# dmabuf_holder -- minimal dma-buf importer
#
# Build against the running kernel (only meaningful on the host that owns
# the GPUs, not inside an unprivileged container):
#   make -C /lib/modules/$(uname -r)/build M=$PWD modules CC=gcc-14
#
# Clean:
#   make -C /lib/modules/$(uname -r)/build M=$PWD clean

obj-m += dmabuf_holder.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

.PHONY: all clean
