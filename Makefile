# barlink-torch -- convenience build targets
#
# make blrun          build the capability launcher (then re-setcap, see below)
# make dmabuf_holder  build the kernel module against the running kernel
# make all            both
# make clean          both
#
# After `make blrun` (binary changed), re-install file caps as root:
#   sudo chown root:root tools/blrun && sudo setcap cap_sys_admin+eip tools/blrun

CC      ?= cc
KCC     ?= gcc-15   # match the compiler that built the running kernel (see /proc/version)
KDIR    ?= /lib/modules/$(shell uname -r)/build

.PHONY: all clean blrun dmabuf_holder

all: blrun dmabuf_holder

blrun: tools/blrun.c
	$(CC) -O2 -Wall -o tools/blrun tools/blrun.c

dmabuf_holder:
	$(MAKE) -C $(KDIR) M=$(CURDIR)/dmabuf_holder modules CC=$(KCC)

clean:
	rm -f tools/blrun
	$(MAKE) -C $(KDIR) M=$(CURDIR)/dmabuf_holder clean
