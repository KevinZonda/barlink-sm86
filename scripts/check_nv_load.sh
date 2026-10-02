#!/bin/bash
nvidia-smi
lsof /dev/nvidia* 2>/dev/null | head
