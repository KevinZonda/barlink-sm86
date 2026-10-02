/* SPDX-License-Identifier: GPL-2.0 */
/*
 * dmabuf_holder -- shared interface between the kernel module and userspace.
 *
 * Purpose: accept a dma-buf fd and run dma_buf_get + dma_buf_attach +
 * dma_buf_map_attachment on it, so that the vendor driver's nv_dma_buf_map()
 * (kernel-open/nvidia/nv-dmabuf.c:1066) programs the buffer's BAR1 pages.
 * The attachment is held until it is explicitly released or the character
 * device's file descriptor is closed.
 *
 * This file is included both by the kernel module and by dmabuf_pcie_probe.cpp,
 * so it must only use kernel UAPI types.
 */

#ifndef _DMABUF_HOLDER_H_
#define _DMABUF_HOLDER_H_

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/ioctl.h>
#else
#include <linux/types.h>
#include <sys/ioctl.h>
#endif

#define DMABUF_HOLDER_DEVICE_NAME "dmabuf_holder"
#define DMABUF_HOLDER_DEVICE_PATH "/dev/" DMABUF_HOLDER_DEVICE_NAME

/*
 * Flags for struct dmabuf_holder_hold.flags
 *
 * DMABUF_HOLDER_F_BDF_VALID must be set. The module requires a real PCI
 * device, because nv_dma_buf_attach() calls to_pci_dev(attachment->dev)
 * without any check (nv-dmabuf.c:1018, :1033, :1034), and nv_dma_map_peer()
 * does the same (nv-dma.c:749). A dummy device that does not embed a
 * struct pci_dev would make those calls read memory outside the object.
 * There is deliberately no dummy-device mode.
 */
#define DMABUF_HOLDER_F_BDF_VALID   (1u << 0)

/* One entry of the returned sg table. */
struct dmabuf_holder_sg_entry {
	__u64 dma_address;	/* sg_dma_address() */
	__u64 dma_len;		/* sg_dma_len()     */
};

struct dmabuf_holder_hold {
	/* Input */
	__s32 dmabuf_fd;	/* the dma-buf fd to hold                      */
	__u32 flags;		/* DMABUF_HOLDER_F_*                           */
	__u32 pci_domain;	/* BDF of the device to attach as              */
	__u8  pci_bus;
	__u8  pci_slot;
	__u8  pci_func;
	__u8  reserved0;
	__u32 max_entries;	/* capacity of the buffer behind 'entries'     */
	__u32 reserved1;
	__u64 entries;		/* userspace address: dmabuf_holder_sg_entry[] */

	/* Output */
	__u32 handle;		/* identifier for RELEASE                      */
	__u32 nents;		/* actual number of sg entries                 */
	__u64 dmabuf_size;	/* size of the dma-buf in bytes                */
	__u64 total_len;	/* sum of all sg_dma_len()                     */
};

struct dmabuf_holder_release {
	__u32 handle;
	__u32 reserved;
};

#define DMABUF_HOLDER_IOC_MAGIC 0xDB

/* HOLD: dma_buf_get + dma_buf_attach + dma_buf_map_attachment */
#define DMABUF_HOLDER_IOC_HOLD \
	_IOWR(DMABUF_HOLDER_IOC_MAGIC, 1, struct dmabuf_holder_hold)

/* RELEASE: dma_buf_unmap_attachment + dma_buf_detach + dma_buf_put */
#define DMABUF_HOLDER_IOC_RELEASE \
	_IOW(DMABUF_HOLDER_IOC_MAGIC, 2, struct dmabuf_holder_release)

#endif /* _DMABUF_HOLDER_H_ */
