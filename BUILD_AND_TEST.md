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

## 7. torch.distributed ProcessGroup 后端（barlink）

`torch_ext/barlink_sm86/process_group.py` 把 peer 模式包装成 torch.distributed 后端，
纯 Python，无需改 C++。注册 + 初始化三行：

```python
from barlink_sm86 import process_group as blpg
blpg.init_process_group(init_method="tcp://127.0.0.1:29500", rank=r, world_size=2)
dist.all_reduce(tensor)        # SUM；fp32/fp64/bf16 原生，fp16 经 fp32 staging
```

```bash
bash torch_ext/tests/run_pg.sh    # 两进程全流程测试（allreduce/broadcast/barrier/复用/错误路径）
```

环境变量：`BL_POOL_MB`（默认 64，PG 测试用 192 跑 96 MiB 分块路径）、`BL_SOCK_PATH`
（默认 `/tmp/barlink-pg-{uid}-{BL_PG_ID|default}.sock`）、`BL_DEVICE` / `LOCAL_RANK`
（选卡，默认 rank）。

限制清单（v1）：

- **world_size 必须是 2**（底层链路双卡专用）；creator 里直接报错
- **allreduce 只支持 ReduceOp.SUM**；dtype 支持 fp32/fp64/bf16/fp16/fp8×2/u8，其余 raise
- **同步语义**：collective 返回 None 表示已排队，完成性由 current stream 的后续
  enqueue/sync 保证，不是 host 同步
- send/recv/all_gather/reduce_scatter 未实现（vLLM/DiT v1 用不到），调用 raise
  NotImplementedError
- subgroup（new_group 小于 world）不支持：后端是单例复用同一条链路，SPMD 纪律要求
  两端调用序列完全一致
- **torchrun 启动拿不到 CAP_SYS_ADMIN**（file cap 不跨 execve 传递），workaround：
  继续用 `tools/blrun` 直启两个进程（`tests/run_pg.sh` 就是模板），不要 torchrun

实测 PG 层 all_reduce 延迟（双 3080，2026-10-03，pool 192 MiB，50 次平均，
raw bl 链路 ≤64 KiB 是 22-26 µs）：

| size | PG all_reduce |
|---|---|
| 10 KB | 92 µs |
| 64 KB | 96 µs |
| 256 KB | 117 µs |
| 1 MB | 205 µs |

PG 层比 raw bl 多出的 ~70 µs 固定开销 = staging 填充/回拷（两次 D2D copy）+
Python/trampoline 调度；带宽项（>256 KB 后每翻倍 +~90 µs ≈ 12 GB/s 链路速度）一致。

## 8. FLUX.2 Klein 9B DiT 张量并行 demo（2026-10-03）

真实模型 TP=2 验证：双进程各持一张卡，diffusers 官方 `_tp_plan` 走
`parallelize_module`，rowwise allreduce 全部经 barlink PG（BAR1 P2P 链路）。
代码在 `demos/flux2_tp/`（`worker.py` 单文件同时支持 `--tp 1/2`，`run.sh`
编排 + 正确性比对，`results/` 存日志与数字）。

模型：`/mnt/modelzoo/black-forest-labs/FLUX.2-klein-base-9B/transformer`
（diffusers 格式，9B bf16 = 16.9 GiB）。只用 transformer 本体做 DiT step
benchmark：固定 seed 随机输入 img 4096 token（64×64 latent, patch=1,
128ch）+ text context 512×12288 bf16 + timestep 0.5，`torch.inference_mode()`
forward，CUDA event 计时，10 步（先 3 步 warmup）。

### 结果

| 配置 | ms/step (mean, min–max) | 峰值显存 |
|---|---|---|
| TP=1（单卡全模型） | 1401.3 (1391.9–1410.3) | 17.7 GiB |
| TP=2（barlink PG） | 927.1 (922.3–929.7) | 16.9 GiB（含切分前的全量副本） |

**加速比 1.51×**（token 数 4096+512，batch 1，bf16，2026-10-03 实测）。
每 step 有 56 次 rowwise allreduce（8 double block × 4 + 24 single block × 1），
通信量 ~1.5 GB/step（最大消息 37.7 MB，double block img 侧 33.5 MB）。
理想算力减半对应 700 ms，实际 927 ms —— 差额 ~230 ms 即通信暴露 +
每 rank 显存减半收益未计入算力（kernel launch、访存模式变化）。

正确性（同 seed 输入，TP1 vs TP2 输出）：

- cosine = 0.9998，rel_L2 = 1.77%，max_abs_err = 0.057（max|ref| = 4.53），
  `allclose(atol=2e-2, rtol=1e-2)` 覆盖 98.1% 元素，逐 token rel_L2 最大 3.0%、
  无离群 token。
- 1% 阈值直接卡不过，但这**不是 bug**：用 anchor 实验（`--compute-fp32`：
  bf16 权重 + fp32 计算的真值参考）测得 bf16 噪声底 —— TP1-bf16 离 anchor
  rel_L2 = 2.31%，TP2-bf16 离 anchor 2.42%，二者互相之间（1.77%）比各自离
  真值更近。**TP=2 与 TP=1 等距于真值，即 TP 切分正确**。
- 判据（bf16 合理带）：cosine > 0.999 且 rel_L2 < 3%。`results/correctness.json`。

### 踩到的三个坑（都修了）

1. `parallelize_module` 默认 `src_data_rank=0`：切分权重时经 mesh scatter/
   broadcast（barlink v1 无 send/recv，直接撞 `NotImplementedError`）。
   两个 rank 加载的是同一份完整 checkpoint，`src_data_rank=None` 即纯本地
   切分（diffusers `_styles` 的 plan 照常传）。
2. **后端 collective 返回 `None` 会让 DTensor 段错误**：RowwiseParallel 输出
   Partial→Replicate 走 `torch.ops._c10d_functional.all_reduce`，C++ 侧对
   返回的 Work 解引用 —— Python 后端返回 None = 空指针，wait 时 SIGSEGV
   （栈看上去像在 dropout，实为 AsyncCollectiveTensor 延迟等待点）。
   修复：`process_group.py` 的 allreduce/broadcast/barrier 返回
   `_CompletedWork`（已完成语义，与 stream-ordered 约定一致）。
3. DTensor 参数做 dtype cast 必须走 `Module.to(dtype)`，`p.data = p.data.to()`
   对 DTensor 不生效（静默留 bf16，forward 才报 dtype mismatch）。
   仅 anchor 实验用到。

### 复现

```bash
bash demos/flux2_tp/run.sh    # TP=1 → TP=2 → 正确性比对，结果落 results/
```

注意：run.sh 会等 GPU0 空闲再跑（本机可能有其它实验在轮流用卡），每阶段
失败自动重试 10 次。环境：diffusers 0.40.0（自带 Flux2 `_tp_plan`），
torch 2.14.0+cu130，`BL_POOL_MB=192`。
