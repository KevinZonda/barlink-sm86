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
// Security model: the child keeps CAP_SYS_ADMIN only from exec until
// barlink_sm86's init() finishes and drops it (see binding.cpp
// dropCapsAfterInit). PR_SET_NO_NEW_PRIVS guarantees the dropped
// capability can NEVER be regained via file-cap exec in this process.
// Only use on a single-user dev box.

#define _GNU_SOURCE
#include <linux/capability.h>
#include <stdio.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
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
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        perror("prctl(PR_SET_NO_NEW_PRIVS)");
        return 1;
    }
    // File caps (eip) land the cap in permitted+effective but NOT in the
    // process inheritable set, and PR_CAP_AMBIENT_RAISE requires it in
    // permitted AND inheritable. Copy permitted -> inheritable first; a
    // process may always add caps from its own permitted set.
    struct __user_cap_header_struct hdr = { _LINUX_CAPABILITY_VERSION_3, 0 };
    struct __user_cap_data_struct data[_LINUX_CAPABILITY_U32S_3];
    if (syscall(SYS_capget, &hdr, data) != 0) {
        perror("capget");
        return 1;
    }
    data[0].inheritable |= data[0].permitted;
    data[1].inheritable |= data[1].permitted;
    if (syscall(SYS_capset, &hdr, data) != 0) {
        perror("capset(inheritable)");
        return 1;
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
