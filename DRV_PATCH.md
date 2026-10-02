# 驱动补丁使用说明

`patch/$DRV_VERSION/BARLINK_PCIE_MINIMAL.patch` 针对 `open-gpu-kernel-modules@$DRV_VERSION`，
只放宽 `osCheckGpuBarsOverlapAddrRange` 守卫（+`BarlinkPeerBar1` regkey），
是小 BAR 双卡 dma-buf P2P 的最小驱动变更。仅支持 Linux。

## 打补丁并编译

```bash
DRV=595.104.02   # 或 595.58.03 / 580.178.04
git clone --depth 1 --branch $DRV https://git.kigml.com/NVIDIA/open-gpu-kernel-modules.git drv/$DRV
cd drv/$DRV
git apply --check ../../patch/$DRV/BARLINK_PCIE_MINIMAL.patch   # 先验证
git apply ../../patch/$DRV/BARLINK_PCIE_MINIMAL.patch
make modules -j$(nproc)   # gcc 需 ≥ 内核构建版本（ubuntu 6.17 用 gcc-13，7.0 用系统默认 gcc-15）
```

## 加载（先卸 stock 驱动，Ampere 必须 PCI FLR）

```bash
rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia    # nvidia_modeset 必须与 nvidia 一起卸
echo 1 > /sys/bus/pci/devices/<BDF>/reset            # 每张卡的 FLR
insmod nvidia.ko NVreg_RegistryDwords="BarlinkPeerBar1=1"
insmod nvidia-uvm/nvidia-uvm.ko
```

注意：只需 `BarlinkPeerBar1=1` 一个 key。**不要**设 `RMForceStaticBar1`（GSP 固件未补丁，小 BAR 卡必挂）。

## 回退

崩溃后冷启动即回 stock；或重 insmod 官方 .ko。

详见 barlink-pcie 仓库的 `docs/BUILDING_THE_DRIVER.md`（1313 行完整手册，含 FLR、容器 CAP_SYS_ADMIN 等全部坑）。
