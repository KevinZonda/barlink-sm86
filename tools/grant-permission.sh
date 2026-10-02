#!/bin/bash
sudo chown root:root blrun
sudo setcap cap_sys_admin+eip blrun
getcap blrun
