
WIN = 0
WIN64 = 0

# sort_engine.so for msieve's GPU polynomial selection.
#
# Rewritten for CUDA >= 12.8 / Blackwell (sm_120):
#   * uses the CUB bundled with the CUDA toolkit (not the vendored
#     sm_20-era copy that used to live under ./cub) so radix sort gets
#     architecture-tuned policies for sm_120;
#   * builds a native sm_120 cubin plus embedded compute_120 PTX (so the
#     driver can still JIT for any future >= 12.0 device);
#   * links the CUDA runtime statically, so the resulting .so depends only
#     on libc/libstdc++/libcuda -- it drops straight into the runtime
#     container with no libcudart to ship.
#
# 'sm' is accepted for backward compatibility with the top-level Makefile
# but the target arch is fixed at sm_120 for this solution.

ifeq ($(WIN),1)
	CUDA_ROOT = $(shell echo $$CUDA_PATH)
	NVCC = "$(CUDA_ROOT)/bin/nvcc"
	EXT = dll
	NVCCFLAGS += -shared
else
	NVCC = "$(shell which nvcc)"
	CUDA_ROOT = $(shell dirname $(NVCC))/../
	EXT = so
	NVCCFLAGS += -shared -Xcompiler -fPIC -Xcompiler -fvisibility=hidden
endif

# native SASS for sm_120 + PTX fallback for forward compatibility
SM_TARGETS += -gencode=arch=compute_120,code=sm_120 \
	      -gencode=arch=compute_120,code=compute_120

# only the toolkit include dir: pick up the modern <cub/cub.cuh>, never
# the stale vendored one. sort_engine.h is included with quotes so it is
# found next to sort_engine.cu without adding '.' to the search path.
INC = -I"$(CUDA_ROOT)/include"

DEPS = ./Makefile \
	sort_engine.cu \
	sort_engine.h

LIBNAME = sort_engine

all: $(LIBNAME).$(EXT)
	touch built

clean :
	rm -f  *.dll *.so *.lib *.exp built

$(LIBNAME).$(EXT) : $(DEPS)
	$(NVCC) $(SM_TARGETS) -o $@ sort_engine.cu $(NVCCFLAGS) $(INC) \
		-cudart static -O3
