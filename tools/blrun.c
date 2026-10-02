// SPDX-License-Identifier: MIT
//
// blrun -- run a python script with barlink_sm86 pre-initialized and all
// capabilities already dropped. Safe replacement for caprun as the normal
// launcher: user code NEVER holds CAP_SYS_ADMIN.
//
// Execution model:
//   blrun (file caps cap_sys_admin+eip, root-owned)
//     -> raise ambient CAP_SYS_ADMIN, set no_new_privs
//     -> exec .venv/bin/python -m barlink_sm86._bootstrap <script> [args..]
//        (sys.path gets the torch_ext dir via PYTHONPATH)
//     -> _bootstrap: import torch_ext, bl.init()  [caps dropped on return]
//     -> runpy.run_path(<script>)                 [payload runs capless]
//
// The capability window covers only fixed, repo-controlled code: the
// python interpreter, torch import, and bl_init(). The user script runs
// with CapEff == 0 and cannot regain any capability (no_new_privs).
//
// Setup (once, as root, after every rebuild):
//   sudo chown root:root tools/blrun && sudo setcap cap_sys_admin+eip tools/blrun
//
// Usage:
//   BL_DEVICES=0,1 BL_POOL_MB=64 tools/blrun train.py --lr 0.01
// Defaults: BL_DEVICES=0,1  BL_POOL_MB=64.

#define _GNU_SOURCE
#include <libgen.h>
#include <limits.h>
#include <linux/capability.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef CAP_SYS_ADMIN
#define CAP_SYS_ADMIN 21
#endif

static int raiseCap(void)
{
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        perror("prctl(PR_SET_NO_NEW_PRIVS)");
        return -1;
    }
    // file caps land in permitted+effective but not inheritable; ambient
    // raise needs the cap in permitted AND inheritable
    struct __user_cap_header_struct hdr = { _LINUX_CAPABILITY_VERSION_3, 0 };
    struct __user_cap_data_struct data[_LINUX_CAPABILITY_U32S_3];
    if (syscall(SYS_capget, &hdr, data) != 0) {
        perror("capget");
        return -1;
    }
    data[0].inheritable |= data[0].permitted;
    data[1].inheritable |= data[1].permitted;
    if (syscall(SYS_capset, &hdr, data) != 0) {
        perror("capset(inheritable)");
        return -1;
    }
    if (prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_RAISE, CAP_SYS_ADMIN, 0, 0) != 0) {
        perror("prctl(PR_CAP_AMBIENT_RAISE)");
        fprintf(stderr, "blrun: setcap missing? run:\n"
                        "  sudo chown root:root tools/blrun\n"
                        "  sudo setcap cap_sys_admin+eip tools/blrun\n");
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: blrun <script.py> [args...]\n"
                        "env: BL_DEVICES=0,1  BL_POOL_MB=64\n");
        return 2;
    }
    if (raiseCap() != 0)
        return 1;

    // locate .venv relative to this executable (tools/ is a subdir of the repo)
    char self[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n < 0) {
        perror("readlink(/proc/self/exe)");
        return 1;
    }
    self[n] = 0;
    char repo[PATH_MAX];
    strncpy(repo, self, sizeof(repo) - 1);
    repo[sizeof(repo) - 1] = 0;
    dirname(repo);          // .../tools
    dirname(repo);          // repo root

    char python[PATH_MAX], pypath[PATH_MAX * 2];
    snprintf(python, sizeof(python), "%s/.venv/bin/python", repo);
    snprintf(pypath, sizeof(pypath), "%s/torch_ext", repo);
    setenv("PYTHONPATH", pypath, 1);

    char **nargv = calloc((size_t)argc + 4, sizeof(char *));
    if (!nargv) {
        perror("calloc");
        return 1;
    }
    nargv[0] = python;
    nargv[1] = (char *)"-m";
    nargv[2] = (char *)"barlink_sm86._bootstrap";
    for (int i = 1; i < argc; ++i)
        nargv[i + 2] = argv[i];
    nargv[argc + 2] = NULL;

    execv(python, nargv);
    perror("execv(.venv/bin/python)");
    fprintf(stderr, "blrun: is the venv present at %s?\n", python);
    return 1;
}
