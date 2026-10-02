# 构建与测试流程

目标：双卡（如 RTX 3080, 256 MiB BAR1）GPU 经 PCIe BAR1 直连 P2P 写，不经主机内存。
路线：MINIMAL 补丁（114 行守卫放宽）+ dma-buf 动态 BAR1 映射。
2026-10-02 实测：双向 `bad_bytes=0`，64 MiB 稳定 13.2 GB/s（stock staging 6.7 GB/s 的 2.2×）。

## 0. 前置条件（一次性）

- Linux + root。内核任意（模块本地编译，但**每次内核升级要重编**，见 §5）
- 驱动分支三选一：`patch/` 下已有 `580.178.04` / `595.58.03` / `595.104.02`
- **AMD 平台必须**：GRUB 加 `iommu=pt`（`sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="iommu=pt /' /etc/default/grub && sudo update-grub && reboot`）。
  默认 AMD-Vi 翻译模式会静默丢弃 peer BAR 写（dmesg 见 `IO_PAGE_FAULT`），这是 GPUDirect 的通用要求

## 1. 构建

```bash
cd barlink-torch
DRV=580.178.04                                   # 选版本
git clone --depth 1 --branch $DRV https://git.kigml.com/NVIDIA/open-gpu-kernel-modules.git drv/$DRV
cd drv/$DRV
git apply ../../patch/$DRV/BARLINK_PCIE_MINIMAL.patch
make modules -j$(nproc)                          # gcc 需 ≥ 内核构建版本
cd ../../..
make -C dmabuf_holder   # 独立 .ko，与 GPU 驱动零依赖
cd bench/bar1-p2p-write
make all-branches                                # 产出 bar1-p2p-write(595) 和 -580
```

验证：`modinfo drv/$DRV/kernel-open/nvidia.ko | grep vermagic` 必须等于 `uname -r`。

## 2. 加载（每次重启后）

```bash
sudo systemctl stop display-manager              # 若 GPU 被桌面占用
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
echo 1 | sudo tee /sys/bus/pci/devices/0000:09:00.0/reset   # 每张卡 FLR，Ampere 必须
echo 1 | sudo tee /sys/bus/pci/devices/0000:0a:00.0/reset
cd ~/projects/nv-p2p/barlink-torch
sudo insmod drv/$DRV/kernel-open/nvidia.ko NVreg_RegistryDwords="BarlinkPeerBar1=1"
sudo insmod drv/$DRV/kernel-open/nvidia-uvm.ko
sudo insmod dmabuf_holder/dmabuf_holder.ko
```

只需 `BarlinkPeerBar1=1` 一个 key。**绝不设** `RMForceStaticBar1`。
验证：`nvidia-smi` 出卡、`cat /proc/driver/nvidia/params | grep -i barlink` 出 1、`/dev/dmabuf_holder` 存在。

### 2b. 免 sudo 运行 bench / torch（一次性）

sudo 被三层东西挡住，前两层是文件权限，第三层是进程 capability：

1. `/dev/dmabuf_holder`：模块写死 0600 → udev 规则解决
2. `/sys/bus/pci/devices/*/resource1_wc`：内核写死 0600，mmap 本身无
   CAP_SYS_RAWIO 检查 → udev 规则 chown+chmod 解决
3. `cudaHostRegister(IoMemory)`：RM 查 `osIsAdministrator()` =
   `capable(CAP_SYS_ADMIN)`（`common/inc/nv-linux.h:537`），与文件权限无关
   → 用 file capabilities 代替 sudo（见下）

```bash
sudo cp udev/99-barlink.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
sudo chown root:kevin /dev/dmabuf_holder && sudo chmod 660 /dev/dmabuf_holder
sudo chown root:kevin /sys/bus/pci/devices/*/resource1_wc && sudo chmod 660 /sys/bus/pci/devices/*/resource1_wc
# 第 3 层：blrun 启动器（torch 脚本，init 后丢权再跑 payload）
sudo chown root:root tools/blrun
sudo setcap cap_sys_admin+eip tools/blrun
# bench 二进制直接 setcap（短寿命本地测试工具，cap 伴随整个运行期）
sudo chown root:root bench/bar1-p2p-write/bar1-p2p-write-* && sudo setcap cap_sys_admin+ep bench/bar1-p2p-write/bar1-p2p-write-*
```

之后 bench 和 torch 全部免 sudo。日常入口用 **blrun**（payload 永不持权）：

```bash
bench/bar1-p2p-write/bar1-p2p-write-580 --both                    # bench
tools/blrun torch_ext/tests/test_basic.py                          # torch 脚本
BL_POOL_MB=128 tools/blrun train.py --lr 0.01                      # 自定义 pool
```

**安全模型**：cap 的生存窗 = python 解释器 + torch import + `bl.init()`，全部是仓库内
固定代码；`bl.init()` 返回即丢权（且 nnp 保证找不回），**用户脚本启动时 CapEff 已是 0**。
即使把恶意 payload 喂给 blrun，它拿到手时也是普通用户权限。

注意：rebuild `tools/blrun` 后需重跑上面两条 chown/setcap；bench 二进制的 cap 伴随
整个运行期（短寿命本地测试工具，可接受）。§2 的 insmod/rmmod 仍需要 root。

## 3. Bench

```bash
cd ~/projects/nv-p2p/barlink-torch/bench/bar1-p2p-write
BIN=./bar1-p2p-write-580                         # 与驱动分支一致
sudo stdbuf -oL -eL $BIN --both 2>&1 | tee trials/<drv>-patched/run.log
```

- `--size`（默认 64M，受 256 MiB BAR1 限制）、`--iters`、`--both`、`--diag`（分层诊断）
- 通过标准：两个方向 Phase 1 `bad_bytes = 0`，Phase 2 出 GB/s
- **不要用 `cuMemcpyDtoH` 做验证判定**（stale L2 假象）；bench 内已用目标卡 `ld.global.cv` kernel 为准

回退：`sudo rmmod dmabuf_holder nvidia_uvm nvidia && sudo modprobe nvidia && sudo modprobe nvidia_uvm`（或冷启动）。

## 4. 数据与基线

- 每次跑存 `bench/bar1-p2p-write/trials/<drv>[-patched]/`（run.log + env.txt + README）
- stock 基线：`bench/stock-d2d/`（跨卡拷贝 staging 带宽，参照 6.7 GB/s）
- 已验证参考值：4 KiB 1.7 / 64 KiB 9.0 / ≥1 MiB 13.2 GB/s，双向对称

## 5. 排错速查

| 症状 | 原因 |
|---|---|
| `insmod` 报 version/symbol 错 | 内核升级后没重编（§1）；或没先 rmmod stock |
| `Invalid parameters` | stock 模块还占着，或 .ko 与内核不匹配 |
| Phase 1 ~100% 错、两次运行逐字节一致 | IOMMU 没开 pt（§0），写被丢弃 |
| `cudaHostRegister` 失败 | 补丁没打 / regkey 没设 / 无 CAP_SYS_ADMIN |
| 有输出但 tee 的 log 空 | 程序崩溃丢了块缓冲 stdout——已用 `stdbuf -oL` 解决；看 `${PIPESTATUS[0]}` 而非 `$?` |
| `NV_ESC_CHECK_VERSION_STR` 失败 | bench 二进制分支与驱动不符，换 `bar1-p2p-write-580/-595` |
| 仅供 stock 的 P2P 基准 | 不会有提升。本补丁只服务 dma-buf BAR1 路径，不解锁 `cudaDeviceCanAccessPeer`/NCCL |

另：卸载映射时驱动可能打 `pIOVAS != NULL` 断言噪音，不影响数据面，忽略。

## 6. Bench 对比（实测）

环境：双 RTX 3080 20 GiB（同 root complex），patched 580.178.04 + iommu=pt，2026-10-02。
BAR1 直连（kernel 写对端 BAR，`--both` 两方向取一）vs stock 580（跨卡拷贝经主机内存 staging，`bench/stock-d2d`）：

| size | BAR1 直连 | stock staging | 提升 |
|---|---|---|---|
| 4 KiB | 1.7 GB/s | 0.47 GB/s | 3.6× |
| 64 KiB | 9.0 GB/s | 2.9 GB/s | 3.1× |
| 1 MiB | 12.8 GB/s | 6.1 GB/s | 2.1× |
| 16 MiB | 13.2 GB/s | 6.2 GB/s | 2.1× |
| 64 MiB | 13.2 GB/s | 6.0 GB/s | 2.2× |

- 双向对称（0→1 与 1→0 相差 <0.5%）；原始数据 `trials/580.178.04-patched/run.log`、基线 `trials/580.178.04/`
- ≥1 MiB 的 13.2 GB/s 是链路平台期（对照 barlink-pcie 在 x8 Gen4 的 12.7 GB/s ≈ 链路 80%），补丁大小不影响带宽
- 提升随尺寸变小而增大：大包省一半 PCIe 往返，小包还省主机往返的固定延迟
- **对 stock CUDA/NCCL 基准（p2pBandwidthLatencyTest、nccl-tests）无提升**——补丁只服务 dma-buf BAR1 路径，不解锁 `cudaDeviceCanAccessPeer`
