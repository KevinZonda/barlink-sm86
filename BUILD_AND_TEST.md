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
dist.all_reduce(tensor)        # SUM；连续对齐的 CUDA tensor 走零拷贝路径
                               # （fp16/bf16/fp32/fp64/fp8），其余经 pool staging
```

底层新原语（peer 模式，直接读写用户 tensor，无 staging 拷贝、无 host sync）：

```python
bl.allreduce_into(out, in)     # out = in + peer_in；in==out（原地）或完全不重叠
                               # fp16 原生（float opmath 加一次舍入，与 torch 一致）
                               # u8 不支持（保持 mod-256 pool 语义，走 allreduce_）
bl.send_into(t, peer_rank)     # 零拷贝 p2p：纯字节搬运，任意 dtype（u8 无 mod-256
bl.recv_into(t, peer_rank)     # 语义）；t 需 16 字节对齐、字节数非零 16 倍数、
                               # ≤ p2p scratch 区（pool/4 − flag tail，调用方分块）
```

p2p 协议（与 allreduce 的 arm 握手不同，三个要点）：

- **scratch 分区**：allreduce 固定偏移在 scratch 区**下半部**，p2p 用**上半部**，
  两协议交错无竞争
- **consumed-receipt**：发送方第 k 次 send 等接收方第 k−1 次 recv 在 move kernel
  读完后回执的 consumed flag（第 1 次免等）；不能用"recv 起始 arm"——同 stream 上
  send 的等待自旋会挡在 recv 的 arm 之前，双方互相等待死锁
- **send 走独立 side-stream**（recv 留在调用方 stream）：双方连续多个 send 再 recv
  时，第二个 send 的 consumed 等待不会在单 stream 上挡住本端 recv；collective
  发起前 join send-stream，防止 add 覆写 pending payload 正在读的 tensor

PG 层 `dist.send/recv/isend/irecv`（torch 2.14 的 Backend 只有 send/recv 一对，
isend/irecv/P2POp 都走它们，isend 只是不 wait 返回的 Work）：tag 接受但不传输
（链路无控制通道，按调用序配对）；对齐且 16 倍数的 CUDA tensor 走零拷贝（**任意
dtype**），其余（非 16 倍数、CPU、非连续）经 pool staging buffer 兜底；dst/src 必须
是 `1 - rank`，world≠2 照常 raise。

```bash
bash torch_ext/tests/run_pg.sh        # 两进程全流程测试（allreduce/broadcast/barrier/复用/错误路径）
bash torch_ext/tests/run_pg_p2p.sh    # p2p 全流程测试（全双工/乒乓/staged 兜底/交错/混合流量）
bash bench/latency/run_pg_lat.sh      # PG 层延迟 bench（10KB/64KB/256KB/1MB allreduce）
bash bench/latency/run_pg_p2p_lat.sh  # PG 层 p2p 延迟 bench（ping-pong 往返 + 全双工交换）
```

环境变量：`BL_POOL_MB`（默认 64，PG 测试用 192 跑 96 MiB 分块路径）、`BL_SOCK_PATH`
（默认 `/tmp/barlink-pg-{uid}-{BL_PG_ID|default}.sock`）、`BL_DEVICE` / `LOCAL_RANK`
（选卡，默认 rank）。

限制清单（v1）：

- **world_size 必须是 2**（底层链路双卡专用）；creator 里直接报错
- **allreduce 只支持 ReduceOp.SUM**；dtype 支持 fp32/fp64/bf16/fp16/fp8×2/u8，其余 raise
- **同步语义**：collective 与 p2p 都返回已完成的 Work（单例），完成性由 current
  stream 的后续 enqueue/sync 保证，不是 host 同步（isend 发出的 buffer 在下一个
  bl 调用/barrier 前不要复用）
- **allreduce 零拷贝路径条件**：CUDA + contiguous + 16 字节对齐指针 + 字节数非零
  16 倍数 + ≤ allreduce scratch 区（pool/4 − flag tail，PG 层自动分块）；不满足
  则回退 pool staging（u8、非连续、非对齐、CPU tensor 都走这条）
- **send/recv 零拷贝路径条件**：同上但**任意 dtype**（纯字节搬运），scratch 区是
  p2p 专用的上半区（pool/4 − flag tail）；tag 不传输，按调用序配对；all_gather /
  reduce_scatter 未实现，调用 raise NotImplementedError
- subgroup（new_group 小于 world）不支持：后端是单例复用同一条链路，SPMD 纪律要求
  两端调用序列完全一致
- **torchrun 启动拿不到 CAP_SYS_ADMIN**（file cap 不跨 execve 传递），workaround：
  继续用 `tools/blrun` 直启两个进程（`tests/run_pg.sh` 就是模板），不要 torchrun

实测 PG 层 all_reduce 延迟（双 3080，2026-10-03，pool 192 MiB，50 次 event 计时
back-to-back；raw bl 链路 ≤64 KiB 是 22-26 µs；`bench/latency/run_pg_lat.sh`）：

| size | 零拷贝前 | 零拷贝后 | 提升 |
|---|---|---|---|
| 10 KB | 92 µs | **25 µs** | 3.7× |
| 64 KB | 96 µs | **23 µs** | 4.2× |
| 256 KB | 117 µs | **43 µs** | 2.7× |
| 1 MB | 205 µs | **129 µs** | 1.6× |

实测 PG 层 p2p 延迟（双 3080，2026-10-03，同方法；`bench/latency/run_pg_p2p_lat.sh`，
`trials/` 有留底）：

| size | ping-pong 往返 | 全双工交换（单向） |
|---|---|---|
| 10 KB | **66 µs** | **29 µs** |
| 64 KB | 66 µs | 29 µs |
| 256 KB | 131 µs | 51 µs |
| 1 MB | 524 µs | 256 µs |

对比：allreduce 10KB 25 µs（双向 payload 同时在飞 + add，摊到单向 ~12 µs）；p2p
全双工交换 29 µs = 单向一次完整投递（send 侧 3 kernel：consumed 等待 + payload +
完成 flag；recv 侧 3 kernel：flag 等待 + move + consumed 回执）。ping-pong 往返
≈ 2 次串行投递（~33 µs/单向），符合链路串行化的预期。1 MB 单向 256 µs ≈
4.1 GB/s（payload + move 各一次 D2D 级拷贝开销，无 add）。

零拷贝协议：单 payload（kernel 直读用户 `in`，st.global.wt 写 peer scratch 固定
偏移）+ 本地 add kernel 直写 `out`（scratch 侧 ld.relaxed.sys 读），arm/完成
flag 协议不变，全程 stream-ordered、无 host sync、无 pool 中转。剩下 ~25 µs =
c10d trampoline + 6 个 kernel launch（arm/armwait/payload/mark/flagwait/add）+
flag 可见性（~3 µs×2）。大消息带宽不退化：32 MB/op 时 8.1 GB/s/方向 = 链路双向
聚合 ~17 GB/s 的单方向份额（平台上限），raw bl 数字不变（`trials/` 有留底：
`c19abd8-pre-zerocopy/` vs `zerocopy/`）。

历史留档：零拷贝前 PG 层比 raw bl 多出的 ~70 µs 固定开销 = staging 填充/回拷
（两次 D2D copy）+ Python/trampoline 调度。

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

| 配置 | ms/step (mean, min–max) | vs TP=1 | 峰值显存 |
|---|---|---|---|
| TP=1（单卡全模型） | 1393.3 (1384.6–1401.7) | 1.00× | 17.7 GiB |
| TP=2 barlink PG（零拷贝） | 896.4 (891.5–905.7) | **1.55×** | 16.9 GiB |
| TP=2 NCCL（SHM 中转，无 P2P） | 918.2 (916.0–922.1) | 1.52× | 16.9 GiB |
| TP=2 gloo（host staging） | 1564.1 | 0.89×（比单卡慢） | 16.9 GiB |

（token 数 4096+512，batch 1，bf16，2026-10-03 实测；零拷贝 PG 前 TP=2-barlink
为 927.1 ms/step = 1.51×，见 git 历史。NCCL/gloo 对比：`bash run.sh nccl|gloo`，
同 seed 同输入，DTensor tp plan 不变，只换 ProcessGroup 后端。）

**结论**：此负载通信带宽主导（~1.5 GB/step，最大消息 37.7 MB），barlink 与
NCCL-SHM 的 step 时间接近（896 vs 918 ms，barlink 快 2.4%），差距主要体现在
小消息固定延迟上（PG 层 10KB：barlink 25 µs vs NCCL-SHM 49 µs，§7）——对
LLM decode 类每 token 多次小 allreduce 的场景意义更大。gloo 在 CUDA tensor 上
可用但走 host 中转，比单卡还慢 11%，不可用。三后端正确性数字完全一致
（cosine 0.9998 / rel_L2 1.77%）：bf16 加法两侧都是"float 加一次舍入"，与
§8 上面测得的 bf16 噪声底一致，即 NCCL/gloo 输出也在真值 2.4% 带内。

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
bash demos/flux2_tp/run.sh           # TP=1 → TP=2-barlink → 正确性比对
bash demos/flux2_tp/run.sh nccl      # TP=2-nccl vs 已存的 TP=1 参考（NCCL_P2P_DISABLE=1）
bash demos/flux2_tp/run.sh gloo      # TP=2-gloo，同上（慢，仅对照）
```

注意：run.sh 会等 GPU0 空闲再跑（本机可能有其它实验在轮流用卡），每阶段
失败自动重试 10 次。环境：diffusers 0.40.0（自带 Flux2 `_tp_plan`），
torch 2.14.0+cu130，`BL_POOL_MB=192`。NCCL 分支每进程用
`CUDA_VISIBLE_DEVICES` 固定一张卡（stock torch，无需 caps）；本机双 3080
无 P2P，NCCL 走 SHM/主机内存中转，init 前设 `NCCL_P2P_DISABLE=1`
`NCCL_IB_DISABLE=1` 跳过能力探测。

## 9. Qwen3.8-27B INT8 W8A16 TP=2 decode：barlink PG vs NCCL（2026-10-04）

模型 `/mnt/modelzoo/lued/Qwen3.8-27B-INT8-W8A16-MTP`：compressed-tensors
pack-quantized INT8 W8A16（group 128 sym），64 层混合 GDN 线性注意力（48）
+ 全注意力（16），hidden 5120，vocab 248320。代码在 `demos/qwen_tp/qwen38_int8/`
（`run.sh all` 一条命令跑完四组 + 正确性比对，结果存 `results/`）。

**int8 实际计算路径**（两 backend 完全相同的分片与计算，第一原则）：checkpoint
的 pack-quantized int32 权重按 **bias-128 字节序**解包（
`packed.view(int8) ^ -128`，与 compressed_tensors unpacker 逐元素验证过），
group-128 bf16 scale 在 fp32 下精确 fold 成 per-row int8，激活做 per-token
动态 int8 量化，matmul 走 **`torch._int_mm`（cublasLt IMMA）**——decode 每
token 每卡只读 1 B/param（int8 shard 11.3 GB）。没有可用的 sm86 W8A16
weight-only kernel（Marlin 只在 vLLM 里；torchao CUDA groupwise 会物化 bf16），
且 bf16 全量反量化 27 GB/卡装不下，故用 W8A8 动态量化替代——两侧一致，
对比有效。`_int_mm` 要求 m>16，decode m=1 时 pad 到 32 行（带宽不变）。
GDN 走 transformers 5.18 原生 torch kernel（无 fla/causal_conv1d，装了就更快，
两边同路径即可）。

**延迟**（greedy，prompt 42 tok，warmup 16，计时 64 tok，双 3080，2026-10-04）：

| backend | tok/s | 通信开销 ms/token |
|---|---|---|
| barlink PG | 6.96（6.63–7.27 三次） | 32.0 |
| NCCL（SHM 中转） | **8.03** | 12.5 |
| 无通信对照（barlink / NCCL） | 8.95 / 8.93 | — |

每 token 通信 = 128 次 rowwise all_reduce（10KB，down/o/out_proj 各一）+ 1 次
lm_head 半 logits 交换（248KB）。摊到单次 all_reduce：barlink ~250 µs vs
NCCL ~98 µs——**这个负载 NCCL 快 ~15%，瓶颈不是链路**（§7 微基准 barlink
10KB allreduce 25 µs < NCCL 49 µs），而是 barlink 协议每 op 6 个 stream-
ordered kernel：decode 是 ~4500 kernel/token 的串行 launch-bound 关键路径，
每 op 多出的 kernel launch 全部落在关键路径上。要翻盘得压 kernel 数
（小消息专用 fused 协议，TODO）。

**正确性**：两 backend 输出**逐位相同**（64 token 全 match，首 token logits
rel_l2 = 0.0；两 rank 加法都在 fp32 域一次舍入，与 NCCL bf16 SUM 的 2-rank
情形逐位一致）。`results/correctness_barlink_vs_nccl.json`。

两个平台坑（都写了备注）：
- torch 2.14 的 NCCL lazy p2p communicator 在本平台**裸 isend/irecv 双卡
  必现 illegal memory access**（与 NCCL_P2P_DISABLE 无关，最小复现 10 行）；
  NCCL 分支的 lm_head 交换改用 all_gather（barlink 分支用 isend/irecv 零拷贝
  p2p；barlink PG 未实现 all_gather）。
- run.sh 的后台管道不能 `| tee`（`$!` 是 tee 的 pid，真实退出码丢失，曾把
  崩溃的 phase 标成 OK）；日志直写文件。
