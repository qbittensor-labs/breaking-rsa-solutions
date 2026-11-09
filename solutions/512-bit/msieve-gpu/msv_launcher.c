// Copyright (C) 2026 qBitTensor Labs.
// Original author: Xdev (Enigma / Breaking RSA competition).
// IP in custom components assigned to qBitTensor Labs under the Enigma rules.
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
// details. You should have received a copy of the license with this program;
// if not, see <https://www.gnu.org/licenses/>.

/* ==========================================================================
 * msv_launcher.c -- memfd broker for the msieve downstream.
 *
 * WHY. The validator gives us a 10 GiB tmpfs on /tmp and 85 GiB of RAM. The
 * relation set is ~8.5 GB and msieve's .mat/.lp intermediates are GB-scale, so
 * writing any of them to /tmp is either impossible or bills the same RAM twice
 * (tmpfs pages count against the cgroup). This process loads the relations into
 * an ANONYMOUS memfd, publishes it as <workdir>/m.dat via a /proc/<pid>/fd
 * symlink, pre-creates empty memfds for the two big outputs msieve writes
 * (m.dat.mat, m.dat.lp), copies the small factor base to <workdir>/m.fb, and
 * then EXECs the downstream script.
 *
 * !! IT MUST exec, NOT fork !!  The symlinks point at /proc/<pid>/fd/<n>, so the
 * pid has to stay the same and the descriptors have to survive the exec. That is
 * why memfd_create is called WITHOUT MFD_CLOEXEC. A forking wrapper would also
 * swallow the SIGTERM that factor_msv.sh's phase-B loop uses to stop the sieve.
 *
 * The kernel frees every memfd when this pid exits, so there is no cleanup path
 * and no way to leak the RAM past the run.
 *
 * usage: msv_launcher <relfile> <fbfile> <workdir> <script> [args...]
 * ========================================================================== */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/sendfile.h>
#include <sys/syscall.h>

#ifndef SYS_memfd_create
#error "SYS_memfd_create is required (Linux >= 3.17)"
#endif

/* No MFD_CLOEXEC -- the fd must survive execvp() so the /proc symlink stays live. */
static int memfd_open(const char *name)
{
    int fd = (int)syscall(SYS_memfd_create, name, 0);
    if (fd < 0) { perror("memfd"); exit(1); }
    return fd;
}

/* Move a descriptor out of the low range and into a fixed slot.
 * !! LOAD-BEARING !!  The memfds are published as /proc/<pid>/fd/<n> symlinks and
 * then inherited across execvp. If one of them sits at fd 3/4/5, the child is free
 * to reuse that number -- a bash redirection (`exec 3>...`) or msieve opening its
 * own files would REPLACE the descriptor the symlink resolves through, and m.dat
 * would silently become the wrong file mid-run. Parking them at 20+ keeps them out
 * of the range anything allocates by accident. (The original broker used 20/21/22;
 * kept identical so the /proc paths in a captured log still line up.) */
static int park(int fd, int slot)
{
    if (fd == slot) return fd;
    if (dup2(fd, slot) < 0) { perror("dup2"); exit(1); }
    close(fd);
    return slot;
}

/* Publish <workdir>/m.dat<suffix> as a symlink to this process's memfd. */
static void publish(int fd, const char *workdir, const char *suffix)
{
    char src[64], dst[4096];
    snprintf(src, sizeof src, "/proc/%d/fd/%d", (int)getpid(), fd);
    snprintf(dst, sizeof dst, "%s/m.dat%s", workdir, suffix);
    unlink(dst);                                  /* stale link from a previous run */
    if (symlink(src, dst) != 0) { perror("symlink"); exit(1); }
}

/* Whole-file copy. sendfile() keeps the 8.5 GB relation copy in the kernel;
   it caps at ~2 GB per call, so loop until the source is drained. */
static off_t copy_all(int in, int out)
{
    off_t total = 0;
    for (;;) {
        ssize_t n = sendfile(out, in, NULL, 1u << 30);
        if (n < 0) { perror("sendfile"); exit(1); }
        if (n == 0) break;
        total += n;
    }
    return total;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s <relfile> <fbfile> <workdir> <script> [args...]\n", argv[0]);
        return 2;
    }
    const char *relfile = argv[1], *fbfile = argv[2], *workdir = argv[3];

    mkdir(workdir, 0755);                         /* pre-existing is fine */

    /* 1. relations -> memfd -> <workdir>/m.dat */
    int rin = open(relfile, O_RDONLY);
    if (rin < 0) { perror(relfile); return 1; }
    int rmem = park(memfd_open("m.dat"), 20);
    off_t nbytes = copy_all(rin, rmem);
    close(rin);
    if (lseek(rmem, 0, SEEK_SET) == (off_t)-1) { perror("lseek"); return 1; }
    publish(rmem, workdir, "");
    fprintf(stderr, "[broker] loaded %ld MB relations into RAM\n", (long)(nbytes >> 20));

    /* 2. empty memfds for the outputs msieve writes, so -nc2 / -nc1 never touch
     *    the tmpfs. m.dat.mat is the matrix; m.dat.lp is the large-prime file. */
    publish(park(memfd_open("m.dat.lp"),  21), workdir, ".lp");
    publish(park(memfd_open("m.dat.mat"), 22), workdir, ".mat");

    /* 3. factor base is small -- a real file, msieve reopens it by name. */
    int fin = open(fbfile, O_RDONLY);
    if (fin < 0) { perror(fbfile); return 1; }
    char fbdst[4096];
    snprintf(fbdst, sizeof fbdst, "%s/m.fb", workdir);
    int fout = open(fbdst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fout < 0) { perror(fbdst); return 1; }
    copy_all(fin, fout);
    close(fin); close(fout);

    /* 4. hand over. execvp keeps the pid, so every symlink above stays valid. */
    char pid[32];
    snprintf(pid, sizeof pid, "%d", (int)getpid());
    setenv("BROKER_PID", pid, 1);

    execvp(argv[4], &argv[4]);
    perror("execvp");
    return 127;
}
