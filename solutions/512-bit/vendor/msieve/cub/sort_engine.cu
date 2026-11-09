/* simple DSO interface for parallel simultaneous sort;
   keys are 32- or 64-bit unsigned integers and values are
   32-bit unsigned integers.

   Built against the CUB that ships with the CUDA toolkit (>= 12.x)
   rather than the ancient sm_20-era CUB that used to be vendored in
   this directory: the in-tree copy carried hand-tuned radix-sort
   policies only up to sm_20/sm_52, so on Blackwell (sm_120) it fell
   back to generic, poorly-occupied kernels. Toolkit CUB auto-selects
   architecture-tuned policies for sm_120, which is the whole point of
   rebuilding this for the RTX PRO 6000. The DSO ABI (sort_data_t and
   the three entry points below) is unchanged, so msieve loads it as
   before. */

#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <cuda.h>
#include <cub/cub.cuh>

#include "sort_engine.h"

typedef unsigned int uint32;

#if defined(_WIN32) || defined (_WIN64)
	#define SORT_ENGINE_DECL __declspec(dllexport)
	typedef unsigned __int64 uint64;
#else
	#define SORT_ENGINE_DECL __attribute__((visibility("default")))
	typedef unsigned long long uint64;
#endif

#define CUDA_TRY(func) \
	{ 			 					\
		cudaError_t status = func;				\
		if (status != cudaSuccess) {				\
			const char * str = cudaGetErrorString(status);	\
			if (!str)					\
				str = "Unknown";			\
			printf("error (%s:%d): %s\n", __FILE__, __LINE__, str);\
			exit(-1);					\
		}							\
	}

struct sort_engine
{
	sort_engine()
	  : temp_data(0), temp_size(0)
	{
	}

	~sort_engine()
	{
		if (temp_size)
			CUDA_TRY(cudaFree(temp_data))
	}

	/* make sure the persistent temp buffer holds at least 'needed'
	   bytes; grows monotonically so a single query up front covers
	   every array in a batch (they are all the same size) */
	void reserve(size_t needed)
	{
		if (needed > temp_size) {
			if (temp_size)
				CUDA_TRY(cudaFree(temp_data))
			CUDA_TRY(cudaMalloc(&temp_data, needed))
			temp_size = needed;
		}
	}

	void * temp_data;
	size_t temp_size;
};

extern "C"
{

SORT_ENGINE_DECL void *
sort_engine_init(void)
{
	return new sort_engine;
}

SORT_ENGINE_DECL void
sort_engine_free(void * e)
{
	delete (sort_engine *)e;
}

SORT_ENGINE_DECL void
sort_engine_run(void * e, sort_data_t * data)
{
	// arrays are assumed packed together; check
	// they would all start on a power-of-two boundary

	if (data->num_arrays > 1 && data->num_elements % 16) {
		printf("sort_engine: invalid array size\n");
		exit(-1);
	}

	sort_engine *engine = (sort_engine *)e;

	if (data->key_bits <= 32) {

		// every array in the batch has the same element count and
		// key width, so the temp-storage requirement is identical;
		// query it once instead of once per array.

		size_t temp_size = 0;
		cub::DoubleBuffer<uint32> probe_keys(
				(uint32 *)data->keys_in,
				(uint32 *)data->keys_in_scratch);
		cub::DoubleBuffer<uint32> probe_values(
				(uint32 *)data->data_in,
				(uint32 *)data->data_in_scratch);
		CUDA_TRY(cub::DeviceRadixSort::SortPairs(
					0, temp_size,
					probe_keys, probe_values,
					data->num_elements,
					0, data->key_bits,
					data->stream))
		engine->reserve(temp_size);

		for (size_t i = 0; i < data->num_arrays; i++) {

			cub::DoubleBuffer<uint32> keys(
					(uint32 *)data->keys_in +
						i * data->num_elements,
					(uint32 *)data->keys_in_scratch +
						i * data->num_elements);

			cub::DoubleBuffer<uint32> values(
					(uint32 *)data->data_in +
						i * data->num_elements,
					(uint32 *)data->data_in_scratch +
						i * data->num_elements);

			// sort for real

			CUDA_TRY(cub::DeviceRadixSort::SortPairs(
						engine->temp_data,
						temp_size,
						keys,
						values,
						data->num_elements,
						0,
						data->key_bits,
						data->stream))

			if (keys.selector)
				std::swap(data->keys_in, data->keys_in_scratch);
			if (values.selector)
				std::swap(data->data_in, data->data_in_scratch);
		}
	}
	else {

		size_t temp_size = 0;
		cub::DoubleBuffer<uint64> probe_keys(
				(uint64 *)data->keys_in,
				(uint64 *)data->keys_in_scratch);
		cub::DoubleBuffer<uint32> probe_values(
				(uint32 *)data->data_in,
				(uint32 *)data->data_in_scratch);
		CUDA_TRY(cub::DeviceRadixSort::SortPairs(
					0, temp_size,
					probe_keys, probe_values,
					data->num_elements,
					0, data->key_bits,
					data->stream))
		engine->reserve(temp_size);

		for (size_t i = 0; i < data->num_arrays; i++) {

			cub::DoubleBuffer<uint64> keys(
					(uint64 *)data->keys_in +
						i * data->num_elements,
					(uint64 *)data->keys_in_scratch +
						i * data->num_elements);

			cub::DoubleBuffer<uint32> values(
					(uint32 *)data->data_in +
						i * data->num_elements,
					(uint32 *)data->data_in_scratch +
						i * data->num_elements);

			// sort for real

			CUDA_TRY(cub::DeviceRadixSort::SortPairs(
						engine->temp_data,
						temp_size,
						keys,
						values,
						data->num_elements,
						0,
						data->key_bits,
						data->stream))

			if (keys.selector)
				std::swap(data->keys_in, data->keys_in_scratch);
			if (values.selector)
				std::swap(data->data_in, data->data_in_scratch);
		}
	}
}

} // extern "C"
