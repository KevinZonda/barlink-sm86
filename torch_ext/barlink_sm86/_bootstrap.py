# SPDX-License-Identifier: MIT
#
# barlink_sm86 bootstrap: init the pool, drop capabilities, then run the
# user script. Invoked by tools/blrun -- user code never holds caps.
# See tools/blrun.c for the security model.

import os
import runpy
import sys

import barlink_sm86 as bl


def main():
    devs = [int(x) for x in os.environ.get("BL_DEVICES", "0,1").split(",")]
    pool_mb = int(os.environ.get("BL_POOL_MB", "64"))
    bl.init(devices=devs, pool_mb=pool_mb)  # drops caps on success

    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
