// SPDX-License-Identifier: MIT
//
// caprun -- run a command with CAP_SYS_ADMIN in its ambient set, without
// sudo. Used for barlink-sm86 bench/torch runs: the patched NVIDIA driver's
// peer-mapping branch (osIsAdministrator() = capable(CAP_SYS_ADMIN)) is the
// only remaining root gate; everything else is solved by udev perms.
//
// Setup (once, as root):
//   sudo chown root:root tools/caprun
//   sudo setcap cap_sys_admin+eip tools/caprun
//
// Usage:  tools/caprun .venv/bin/python tests/test_basic.py
//
// Security: any command run through caprun gets CAP_SYS_ADMIN (loadable
// modules, mounts, ...). Only use on a single-user dev box.

#define _GNU_SOURCE
#include <stdio.h>
#include <sys/prctl.h>
#include <unistd.h>

#ifndef CAP_SYS_ADMIN
#define CAP_SYS_ADMIN 21
#endif

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: caprun <cmd> [args...]\n");
        return 2;
    }
    if (prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_RAISE, CAP_SYS_ADMIN, 0, 0) != 0) {
        perror("prctl(PR_CAP_AMBIENT_RAISE)");
        fprintf(stderr, "caprun: setcap missing? run:\n"
                        "  sudo chown root:root tools/caprun\n"
                        "  sudo setcap cap_sys_admin+eip tools/caprun\n");
        return 1;
    }
    execvp(argv[1], &argv[1]);
    perror("execvp");
    return 1;
}
