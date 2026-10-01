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

NVCC      := $(CUDA_HOME)/bin/nvcc
NVCCFLAGS := -O3 -std=c++14 -lineinfo $(GENCODE)
LDFLAGS   := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs

TARGET := bar1-p2p-write

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS) -lcuda

clean:
	rm -f $(TARGET)

.PHONY: all clean
