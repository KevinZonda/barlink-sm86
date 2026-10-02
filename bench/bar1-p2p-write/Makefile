# bar1-p2p-write -- native BAR1 P2P write benchmark (see main.cu header)

CUDA_HOME ?= /usr/local/cuda

# Architecture: Ampere (RTX 3080) by default. Override with ARCH=89 or
# TORCH_CUDA_ARCH_LIST="8.6;9.0" (first entry wins).
ifneq ($(TORCH_CUDA_ARCH_LIST),)
override ARCH := $(firstword $(subst ;, ,$(TORCH_CUDA_ARCH_LIST)))
endif
ARCH ?= 86
override ARCH := $(subst .,,$(ARCH))

GENCODE := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

# Driver branch the binary is built for (selects the fallback RM version
# string and names the output). The RM ioctl ABI itself is byte-identical
# between 580 and 595; the branch only matters for the
# NV_ESC_CHECK_VERSION_STR handshake string.
BRANCH ?= 595

ifeq ($(BRANCH),595)
TARGET := bar1-p2p-write
else ifeq ($(BRANCH),580)
TARGET := bar1-p2p-write-580
else
$(error BRANCH must be 580 or 595)
endif

NVCC      := $(CUDA_HOME)/bin/nvcc
NVCCFLAGS := -O3 -std=c++14 -lineinfo $(GENCODE) -DDRV_BRANCH=$(BRANCH)
LDFLAGS   := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS) -lcuda

# build both branch binaries
all-branches:
	$(MAKE) BRANCH=595
	$(MAKE) BRANCH=580

clean:
	rm -f bar1-p2p-write bar1-p2p-write-580

.PHONY: all all-branches clean
