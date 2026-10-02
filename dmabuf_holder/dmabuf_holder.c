// SPDX-License-Identifier: GPL-2.0
/*
 * dmabuf_holder -- minimal dma-buf importer.
 *
 * Job: do exactly what only ibv_reg_dmabuf_mr() on an RDMA card used to do,
 * namely create and hold a dma-buf attachment.
 *
 * Background (verified against the driver tree this project patches):
 *   - Without static_phys_addrs, the vendor driver only programs the BAR1
 *     pages of an exported buffer in nv_dma_buf_map()
 *     (kernel-open/nvidia/nv-dmabuf.c:1066, wired up in .map_dma_buf at :1455).
 *   - It is torn down in nv_dma_buf_unmap() (nv-dmabuf.c:1133).
 *   - An open dma-buf fd alone triggers neither. So the fd alone is not
 *     enough; it needs an importer.
 *
 * Why a real PCI device and not a dummy:
 *   nv_dma_buf_attach() calls to_pci_dev(attachment->dev) without any check
 *   and reads its dma_mask (nv-dmabuf.c:1033, :1034); likewise in the
 *   FORCE_PCIE branch (:1018). to_pci_dev() is a container_of() -- for a
 *   struct device that is not embedded in a struct pci_dev, the result
 *   points outside the object. Further down, nv_dma_map_peer() does the same
 *   and accesses peer_pci_dev->resource[] and pci_bus_address()
 *   (nv-dma.c:749, :763, :789). A dummy device is therefore ruled out; the
 *   module requires a BDF.
 *
 * The sg table itself is not used by this module -- it is only held
 * (because dma_buf_unmap_attachment needs it later) and its addresses are
 * reported upward. The side effect that matters is solely the run through
 * nv_dma_buf_map().
 *
 * On the addresses: nv_dma_buf_map_pfns() (nv-dmabuf.c:905) stores the
 * return value of nv_dma_map_peer() (:961) in sg_dma_address(). Its input
 * value is priv->handles[].memArea.pRanges[].start, and nv_dma_map_peer
 * explicitly checks that this value lies within the exporting GPU's BAR1
 * (nv-dma.c:763-778). Without a translating IOMMU (iommu=pt) the value is
 * passed through unchanged, so sg_dma_address() is directly the BAR1
 * address. With a translating IOMMU, dma_map_resource() returns an IOVA
 * that does not correspond to the BAR1 address. Which case applies can
 * only be decided at runtime: userspace compares the reported address
 * against the target card's BAR1 range from sysfs.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/version.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/list.h>
#include <linux/uaccess.h>
#include <linux/dma-buf.h>
#include <linux/dma-mapping.h>
#include <linux/pci.h>
#include <linux/scatterlist.h>

#include "dmabuf_holder.h"

#define DH_PFX "dmabuf_holder: "

static int max_print = 16;
module_param(max_print, int, 0644);
MODULE_PARM_DESC(max_print,
	"How many sg entries per attachment are reported via printk "
	"(0 = none, negative = all). Default 16.");

/* One held attachment. */
struct dh_hold {
	struct list_head          node;
	u32                       handle;
	struct dma_buf           *dmabuf;
	struct dma_buf_attachment *attach;
	struct sg_table          *sgt;
	struct pci_dev           *pdev;
};

/* Per open file descriptor of the character device. */
struct dh_file {
	struct mutex     lock;
	struct list_head holds;
	u32              next_handle;
};

/* ------------------------------------------------------------------ */

static struct sg_table *dh_map(struct dma_buf_attachment *attach)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 2, 0)
	return dma_buf_map_attachment_unlocked(attach, DMA_BIDIRECTIONAL);
#else
	return dma_buf_map_attachment(attach, DMA_BIDIRECTIONAL);
#endif
}

static void dh_unmap(struct dma_buf_attachment *attach, struct sg_table *sgt)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 2, 0)
	dma_buf_unmap_attachment_unlocked(attach, sgt, DMA_BIDIRECTIONAL);
#else
	dma_buf_unmap_attachment(attach, sgt, DMA_BIDIRECTIONAL);
#endif
}

/* Fully tears down an attachment. Must not be called without dh_file->lock
 * held -- the caller either holds the lock or is the sole owner (release
 * path). */
static void dh_hold_destroy(struct dh_hold *h)
{
	if (h->sgt && h->attach)
		dh_unmap(h->attach, h->sgt);
	if (h->attach && h->dmabuf)
		dma_buf_detach(h->dmabuf, h->attach);
	if (h->dmabuf)
		dma_buf_put(h->dmabuf);
	if (h->pdev)
		pci_dev_put(h->pdev);

	pr_info(DH_PFX "Handle %u released.\n", h->handle);
	kfree(h);
}

/* ------------------------------------------------------------------ */

static long dh_ioctl_hold(struct dh_file *df, void __user *uarg)
{
	struct dmabuf_holder_hold arg;
	struct dmabuf_holder_sg_entry __user *uent;
	struct dh_hold *h;
	struct scatterlist *sg;
	unsigned int devfn, i, nents = 0, printed = 0;
	u64 total = 0;
	long ret;

	if (copy_from_user(&arg, uarg, sizeof(arg)))
		return -EFAULT;

	if (arg.dmabuf_fd < 0)
		return -EINVAL;

	if (!(arg.flags & DMABUF_HOLDER_F_BDF_VALID)) {
		pr_err(DH_PFX
		       "HOLD rejected without a BDF. dma_buf_attach needs a "
		       "real PCI device here, because nv_dma_buf_attach() "
		       "calls to_pci_dev(attachment->dev) without a check "
		       "(nv-dmabuf.c:1033). There is deliberately no dummy "
		       "device.\n");
		return -EINVAL;
	}

	h = kzalloc(sizeof(*h), GFP_KERNEL);
	if (!h)
		return -ENOMEM;

	devfn = PCI_DEVFN(arg.pci_slot, arg.pci_func);
	h->pdev = pci_get_domain_bus_and_slot(arg.pci_domain, arg.pci_bus, devfn);
	if (!h->pdev) {
		pr_err(DH_PFX "PCI device %04x:%02x:%02x.%u not found.\n",
		       arg.pci_domain, arg.pci_bus, arg.pci_slot, arg.pci_func);
		ret = -ENODEV;
		goto err;
	}

	h->dmabuf = dma_buf_get(arg.dmabuf_fd);
	if (IS_ERR(h->dmabuf)) {
		ret = PTR_ERR(h->dmabuf);
		h->dmabuf = NULL;
		pr_err(DH_PFX "dma_buf_get(fd=%d) failed: %ld\n",
		       arg.dmabuf_fd, ret);
		goto err;
	}

	h->attach = dma_buf_attach(h->dmabuf, &h->pdev->dev);
	if (IS_ERR(h->attach)) {
		ret = PTR_ERR(h->attach);
		h->attach = NULL;
		pr_err(DH_PFX "dma_buf_attach(dev=%s) failed: %ld "
		       "(-ENOTSUPP means: the exporter rejects the PCI "
		       "topology, see nv_grdma_pci_topology_supported in "
		       "nv-pci.c:2708)\n", pci_name(h->pdev), ret);
		goto err;
	}

	/* This is the exact call that triggers nv_dma_buf_map() and programs
	 * the BAR1 pages. That is the module's sole purpose. */
	h->sgt = dh_map(h->attach);
	if (IS_ERR_OR_NULL(h->sgt)) {
		ret = IS_ERR(h->sgt) ? PTR_ERR(h->sgt) : -EIO;
		h->sgt = NULL;
		pr_err(DH_PFX "dma_buf_map_attachment(dev=%s) failed: "
		       "%ld\n", pci_name(h->pdev), ret);
		goto err;
	}

	/* Report the sg table. We do not use it ourselves. */
	uent = (struct dmabuf_holder_sg_entry __user *)
		(uintptr_t)arg.entries;

	pr_info(DH_PFX "Attachment to %s established, dma-buf size %llu, "
		"sgt->nents=%u orig_nents=%u\n",
		pci_name(h->pdev), (unsigned long long)h->dmabuf->size,
		h->sgt->nents, h->sgt->orig_nents);

	for_each_sgtable_dma_sg(h->sgt, sg, i) {
		u64 addr = (u64)sg_dma_address(sg);
		u64 len  = (u64)sg_dma_len(sg);

		total += len;

		if (max_print < 0 || printed < (unsigned int)max_print) {
			pr_info(DH_PFX "  sg[%u] dma_address=0x%llx len=0x%llx\n",
				nents, (unsigned long long)addr,
				(unsigned long long)len);
			printed++;
		}

		if (uent && nents < arg.max_entries) {
			struct dmabuf_holder_sg_entry e;

			e.dma_address = addr;
			e.dma_len     = len;
			if (copy_to_user(&uent[nents], &e, sizeof(e))) {
				ret = -EFAULT;
				dh_unmap(h->attach, h->sgt);
				h->sgt = NULL;
				goto err;
			}
		}
		nents++;
	}

	if (max_print != 0 && printed < nents)
		pr_info(DH_PFX "  ... %u further sg entries suppressed "
			"(max_print=%d)\n", nents - printed, max_print);

	mutex_lock(&df->lock);
	h->handle = ++df->next_handle;
	list_add_tail(&h->node, &df->holds);
	mutex_unlock(&df->lock);

	arg.handle      = h->handle;
	arg.nents       = nents;
	arg.dmabuf_size = h->dmabuf->size;
	arg.total_len   = total;

	if (copy_to_user(uarg, &arg, sizeof(arg))) {
		/* The handle already exists but userspace has no way to
		 * learn it, so tear it down again immediately. */
		mutex_lock(&df->lock);
		list_del(&h->node);
		mutex_unlock(&df->lock);
		dh_hold_destroy(h);
		return -EFAULT;
	}

	pr_info(DH_PFX "Handle %u holds fd=%d on %s, %u sg entries, "
		"total 0x%llx bytes.\n",
		h->handle, arg.dmabuf_fd, pci_name(h->pdev), nents,
		(unsigned long long)total);

	return 0;

err:
	if (h->attach && h->dmabuf)
		dma_buf_detach(h->dmabuf, h->attach);
	if (h->dmabuf)
		dma_buf_put(h->dmabuf);
	if (h->pdev)
		pci_dev_put(h->pdev);
	kfree(h);
	return ret;
}

static long dh_ioctl_release(struct dh_file *df, void __user *uarg)
{
	struct dmabuf_holder_release arg;
	struct dh_hold *h, *tmp, *found = NULL;

	if (copy_from_user(&arg, uarg, sizeof(arg)))
		return -EFAULT;

	mutex_lock(&df->lock);
	list_for_each_entry_safe(h, tmp, &df->holds, node) {
		if (h->handle == arg.handle) {
			list_del(&h->node);
			found = h;
			break;
		}
	}
	mutex_unlock(&df->lock);

	if (!found)
		return -ENOENT;

	dh_hold_destroy(found);
	return 0;
}

static long dh_unlocked_ioctl(struct file *filp, unsigned int cmd,
			      unsigned long a)
{
	struct dh_file *df = filp->private_data;
	void __user *uarg = (void __user *)a;

	switch (cmd) {
	case DMABUF_HOLDER_IOC_HOLD:
		return dh_ioctl_hold(df, uarg);
	case DMABUF_HOLDER_IOC_RELEASE:
		return dh_ioctl_release(df, uarg);
	default:
		return -ENOTTY;
	}
}

static int dh_open(struct inode *inode, struct file *filp)
{
	struct dh_file *df;

	df = kzalloc(sizeof(*df), GFP_KERNEL);
	if (!df)
		return -ENOMEM;

	mutex_init(&df->lock);
	INIT_LIST_HEAD(&df->holds);
	filp->private_data = df;
	return 0;
}

/* Release everything when the file is closed -- a crashed userspace process
 * leaves nothing behind this way. */
static int dh_release(struct inode *inode, struct file *filp)
{
	struct dh_file *df = filp->private_data;
	struct dh_hold *h, *tmp;

	list_for_each_entry_safe(h, tmp, &df->holds, node) {
		list_del(&h->node);
		pr_info(DH_PFX "close(): cleaning up handle %u.\n", h->handle);
		dh_hold_destroy(h);
	}

	mutex_destroy(&df->lock);
	kfree(df);
	return 0;
}

static const struct file_operations dh_fops = {
	.owner          = THIS_MODULE,
	.open           = dh_open,
	.release        = dh_release,
	.unlocked_ioctl = dh_unlocked_ioctl,
	.compat_ioctl   = compat_ptr_ioctl,
	/* no .llseek: no_llseek was removed in 6.12, the default for
	 * character devices is sufficient here. */
};

static struct miscdevice dh_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name  = DMABUF_HOLDER_DEVICE_NAME,
	.fops  = &dh_fops,
	.mode  = 0600,
};

static int __init dh_init(void)
{
	int rc = misc_register(&dh_misc);

	if (rc) {
		pr_err(DH_PFX "misc_register failed: %d\n", rc);
		return rc;
	}
	pr_info(DH_PFX "loaded, device %s\n", DMABUF_HOLDER_DEVICE_PATH);
	return 0;
}

static void __exit dh_exit(void)
{
	misc_deregister(&dh_misc);
	pr_info(DH_PFX "unloaded.\n");
}

module_init(dh_init);
module_exit(dh_exit);

/* The dma_buf_* symbols live in the DMA_BUF symbol namespace. Since 6.13
 * MODULE_IMPORT_NS expects a string; before that, a bare token. */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 13, 0)
MODULE_IMPORT_NS("DMA_BUF");
#else
MODULE_IMPORT_NS(DMA_BUF);
#endif

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Minimal dma-buf importer: holds dma_buf_map_attachment");
MODULE_VERSION("1.0");
