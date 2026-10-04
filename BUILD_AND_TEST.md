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

| size | 零拷贝前 | 零拷贝后 | fused 小消息协议（≤64KB） |
|---|---|---|---|
| 10 KB | 92 µs | 25 µs | **16.4 µs** |
| 64 KB | 96 µs | 23 µs | **19.4 µs** |
| 256 KB | 117 µs | 43 µs | 43 µs（旧路径） |
| 1 MB | 205 µs | 129 µs | 129 µs（旧路径） |

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
flag 协议不变，全程 stream-ordered、无 host sync、无 pool 中转。

**fused 小消息协议**（payload ≤ 64KB；2026-10-04，当日升级为 **replay-safe
设备计数器协议，3 个 kernel**）：CUDA graph 会不经 host 重放捕获的 kernel，
任何烘焙进 kernel 的 host 序号都会在重放时破坏 flag 单调性（stale flag →
读到旧 payload / 等待超时 trap —— 这就是 vLLM graph 捕获失败的根因）。
现协议完全无 host 序号：K0 单线程把两个期望计数器 +1；K1 各 block 轮询
receipt ≥ 期望−1 后写 payload，末 block（本地原子选举）用
`red.global.add.u64` 给对端 flag +1；K2 各 block 等 flag ≥ 期望后做
add，末 block 给对端 receipt +1。远端标记用 BAR 原子（探针证实功能正确），
发射即忘、无完成延迟暴露。等待链在 op 1 终止。
- 专用 scratch 分区（allreduce 半区的顶部切片）+ 专用 flag/receipt slot，
  fused 与旧 arm 路径**内存不相交**，混合尺寸序列无需跨协议门控（soak：
  300 次验证 op + 50 对 fused/旧路径交错，逐位一致）
- payload 与完成 flag 同 kernel：每 block `__threadfence_system` 后原子
  加本地计数器，最后到达的 block 写 peer flag——与 6-kernel 版
  "kernel 边界 + 单源 PCIe 保序"同一类论证，浸泡验证
- >64KB 走原 6-kernel 路径不变；binding/PG 接口不变
- BAR 原子探针结论：`atom.global.add.u64` 对 peer BAR1 窗口**功能可用**
  （计数逐位正确），但比 posted-write+flag 慢（毫秒级 vs 微秒级），弃用
  （`bl.bar_atomic_probe` 诊断保留）

剩下 ~25 µs =
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
| barlink PG | **8.03**（早期 6.63–7.27 为共卡污染，已修正） | 12.8 |
| NCCL（SHM 中转） | **8.03** | 12.5 |
| 无通信对照（barlink / NCCL） | 8.95 / 8.93 | — |

每 token 通信 = 128 次 rowwise all_reduce（10KB，down/o/out_proj 各一）+ 1 次
lm_head 半 logits 交换（248KB）。摊到单次 all_reduce：两 backend 都是
~95–105 µs。**干净卡下两 backend 平手**（通信开销 12.8 vs 12.5 ms/token；
最初测得的"NCCL 快 15%"是另一会话在卡上轮流跑 15.4GB 任务的共卡污染——其
PCIe/IOMMU 流量不对称放大了 barlink 跨卡 flag 轮询延迟，那片 OOM 重试日志
为证）。通信开销占 decode（~122ms）约 10%；其余是无通信对照也存在的固定
成本：profiler 实测每 token ~29k 个 CUDA kernel、96ms device 时间，其中
400 个 `_int_mm` 只有 21ms，~24k 个小 elementwise/copy kernel（动态量化
wrapper 每 Linear ~10 个 + HF glue）才是主体。

### with fla kernels（2026-10-04 补测）

装上 `causal-conv1d 1.7.0`（源码编译 sm86 CUDA 扩展）与
`flash-linear-attention 0.5.2` 后，transformers 自动走优化 kernel（回退
警告消失）。正确性仍然逐位一致（fla vs 原生 rel_l2 = 0.0，barlink vs NCCL
rel_l2 = 0.0）。结果（`QWEN38_TAG_SUFFIX=_fla`，与旧结果并存 results/）：

| 环境 | barlink PG | NCCL | 无通信对照 | barlink 通信开销 |
|---|---|---|---|---|
| 原生 torch GDN | 8.03 | 8.03 | 8.95 / 8.93 | 12.8 ms/token |
| fla 优化 GDN | 7.95（7.82–7.98） | 8.11（8.02–8.11） | 8.80 / 8.97 | 13.2 ms/token |
| fused 小消息协议（≤64KB） | 8.11 | 8.01 | 8.83 / 8.99 | 10.0 ms/token |
| **+ W8A16 weight-only kernel**（10-04） | **12.57** | **12.36** | — | — |

**W8A16 weight-only int8 kernel**（`torch_ext/barlink_sm86/w8a16.cu`，替换
per-row fold + 动态量化 + `_int_mm` 的 ~10 kernel/Linear 自研路径）：权重保持
checkpoint 原样的 group-128 int8 + fp32 scale（**不再二次量化，精度只升不降**），
加载时转置分块 `[K/16,N,16]`（warp 内 16B/lane 全 coalesced）+ scale 转置
`[G,N]`；激活 bf16 per-token 量化一个 kernel；GEMM thread-per-row、`__dp4a`、
K 按 group 切 S 片（S>1 走 fp32 workspace + atomicAdd + cast）；M 模板
{1,4,16} 覆盖 decode。每 Linear 2–3 kernel。正确性：全 decode shape（含
lm_head 的 124160×5120）vs fp64 参考 rel_l2 ≈ 0.00165（per-token int8 量化
误差底），S/M 全组合通过（`torch_ext/tests/test_w8a16.py`）。端到端 decode
66.8 ms/token（原 122–132），tok/s **+55%**（barlink 8.11→12.57，NCCL
8.01→12.36，干净卡）。与旧路径 token 序列在首 token 后分叉（双量化 vs 精确
group 权重的 ~1% logit 差 → near-tie 翻转），两侧输出均连贯正确；barlink vs
NCCL 的 step-0 logits 仍逐位一致（S>1 的 atomic 顺序引入 ±1ulp 级差异，后续
greedy 漂移）。`QWEN38_W8A16=0` 回退旧路径。lm_head 默认仍 bf16（保持
checkpoint 精度，kernel 已覆盖其 shape）。下一步富矿：decode step 的 CUDA
graph 捕获（barlink fused 协议已 replay-safe，自定义 harness 可直接套）。

fla 没带来端到端提速：GDN 原生路径只占每 token kernel 数的 ~13%（32.5k →
28.7k，省 ~5ms device 时间，被 ±5% 运行噪声淹没）。瓶颈在量化 wrapper 与
HF glue 的 kernel 密度，不在 GDN。**对 fused 小消息协议优先级的启示：不迫
切**——两 backend 通信开销无差，压 barlink 协议 kernel 数（6→3）收益有限；
更大的富矿是真 W8A16 weight-only kernel（Marlin 类）和量化/反量化 fuse。

### CUDA graph 捕获 decode step（2026-10-04，`QWEN38_GRAPH=1`）

W8A16 路径上 decode step 整段捕获进一张 CUDA graph（`demos/qwen_tp/qwen38_int8/worker.py`
`gen()` 的 graph 分支）。方案与结论：

**捕获方案**：
- **StaticCache**（transformers 5.18，`max_cache_len=n_prompt+n+8`）：全
  in-place 更新、地址静态（HF 注释明写 "to preserve the static address for
  cudagraphs"）。DynamicCache 的 `torch.cat` 每步新分配，graph 不安全。
- **显式 `position_ids` 输入 tensor**（`spos`）+ 单 token `input_ids` 静态
  buffer（`sid`）：每步只 `copy_`/`fill_` 输入值，graph 不烘焙数值。
- eager i=1..3（兼作 triton/cublas/protocol warmup），**i==3 捕获
  `[lm → lm_head → argmax]`**（捕获只录制不执行，cache 状态保持诚实）；
  i≥4 `graph.replay()`。注意捕获后 graph 输出 tensor 尚未执行，i==4 的
  输入必须是 eager 上一步的真实 token（曾因此踩过 replay 吃脏输入的
  off-by-one，输出整流错位一格）。
- **通信协议**：per-layer fused allreduce（b3cc1e9 设备计数器协议）与
  lm_head isend/irecv p2p 全部录进 graph。捕获期间两个修复：(1) allreduce
  计时 hook 的 cudaEventCreate 关闭（`_ar_events_off()`）；(2) **p2p 协议
  从 host 序号改为设备计数器**（见下"踩坑"）。

**性能**（干净卡双 3080，64 token 计时，2026-10-04）：

| 模式 | barlink PG | NCCL |
|---|---|---|
| eager 基线 | 12.57–12.68 tok/s（decode ~66–67 ms/token） | 12.36–12.92 |
| **CUDA graph** | **22.1–23.0 tok/s（decode 3.1 ms/token）** | **22.4** |

per-token ~21×（3.09 vs 67 ms），tok/s +75~83%。graph 把 ~32.5k kernel/
token 的 host 启动开销（eager decode 的绝对主体）压缩到一次
`cudaGraphLaunch`。

**正确性**（greedy 逐 token 对比，`results/genids_*_rank0.pt`）：
- **graph vs 同 cache eager（StaticCache + 全 eager，
  `QWEN38_GRAPH_NOCAP=1`）：多次运行 64/64 逐位一致**（gpw2、gfull1），
  其余运行仅 1 个 near-tie token 漂移（如 token 60/63 单点翻转后整流
  分叉）。
- **eager 自身同样偶发单点 near-tie 漂移**（statice 三连跑：se1 逐位
  一致、se2 在 53、se3 在 26 翻转）——该漂移是 barlink fused 协议已存在
  的罕见竞态（~1/1–3k 次 op 交付略陈旧数据，幅度 ~near-tie 量级），
  **与 graph 捕获无关**：NCCL backend graph 与 NCCL StaticCache eager
  64/64 逐位一致，证明 harness 捕获链路本身逐位正确。
- DynamicCache vs StaticCache（均 eager）：token 12 near-tie 翻转后分叉
  （预分配零填充 KV + 不同 mask shape 的 ±ulp 差），两侧输出均连贯。
- 旧 p2p 协议（host 序号）的 eager 流与以上全部不同（61/64 分叉）——
  旧协议在 lm_head 交换上系统性交付陈旧数据，新协议修复后消失。

**踩的坑（都修了）**：
1. **p2p 协议的 host 序号不能进 graph**（本次主要 bug）：旧
   `bl_send_into_peer/bl_recv_into_peer` 在 call 时取 host 序号烘焙进
   kernel（`k_mark(flag, seq)` / `k_flag_wait(flag, seq)`）；graph replay
   重放同一序号 → 接收端 flag 已 ≥ seq（上一步留下的）→ **不等待直接读
   scratch**，跨卡竞争下偶读到上一步的 peer 半 logits。症状：graph 输出
   流整体错位一格（graph[i]==eager[i-1]），触发位置不固定（首次 replay
   或第 4 次 replay 都可能），`QWEN38_GRAPH_SYNC=1` 每步同步无效（序号
   是烘焙的）。修复：照 fused allreduce 的设备计数器模式重写 p2p
   （`k_p2p_bump/k_p2p_send/k_p2p_recv`：本地期望计数器 + 远端 red.add
   累加 flag/receipt + last-block 选举；eager side-stream 路径保留 event
   join，与 graph noSide 路径共用同一套计数器协议）。修复后 piecewise
   （lm_head 交换留在 graph 外）与 full-graph 的漂移率相同，且与 eager
   相同 → 残留罕见竞态定位到 fused allreduce（见下）。
2. **残留罕见竞态（已知问题，未修）**：fused allreduce 协议（b3cc1e9）在
   ~1/1–3k 次 op 交付略陈旧数据（eager paced 下也复现：三连跑 2/3 出现
   单点 near-tie 翻转；`BL_AR_NO_FUSED=1` 走 6-kernel arm 路径三连跑逐位
   一致 → 竞态在 fused 3-kernel 协议内部）。vLLM piecewise graph 下
   allreduce 在 graph 外 eager 执行，从未暴露此 paced 场景。不影响输出
   连贯性，后续单独排查。
3. 捕获期 `dist.isend/irecv` 的 side-stream event record/wait 跨捕获流
   非法 → core.cu 在 `cudaStreamIsCapturing` 时自动走 caller-stream
   noSide 路径（修复 1 之后 noSide 就是常规路径之一，事件 join 仅在
   非捕获的 side-stream 路径使用）。

**回归**（新 p2p 协议后全绿）：`torch_ext/tests/run_pg.sh`、
`run_pg_p2p.sh`、`test_w8a16.py`（ALL OK）、vLLM piecewise graph 冒烟
（Qwen2-1.5B TP=2，输出连贯）。

模型 `/mnt/modelzoo/Qwen/Qwen3.5-27B-GPTQ-Int4`（hf-mirror 被限速到 ~116 KB/s
不可用时走 ModelScope）。混合架构：64 层 = 48 GatedDeltaNet 线性注意力 +
16 全注意力，hidden 5120，vocab 248320。**checkpoint 只有 MLP 是 GPTQ int4**
（qweight v1 布局 [in/8, out]，group 128 sym，desc_act=False），其余 bf16
——bf16 部分单卡 20 GB 放不下。代码 `demos/qwen_tp/`（worker.py + 自写
int4linear.py，run.sh 三模式，results/ 存档）。

**实现**（stock GPTQ 路径在本栈全灭：HF quantizer 丢弃 checkpoint 的
dynamic 排除项把 attention 转成随机初始化的 QuantLinear；gptqmodel 原生
loader 单卡 >20 GB；CPU 后端 torch_aten 每次 forward 反量化）：
- 完全手动加载：meta 设备建骨架 + safetensors 逐 tensor 拉取（跳过
  vision/mtp），量化器整体绕开
- MLP：checkpoint 原始 int4 张量直接喂自写 Triton int4 GEMV（v1 布局原样，
  TP 切分在 group 边界上 = 量化值的精确子集；BN128/BK256/SPLIT2 调优后
  544 GB/s eff，71% HBM 峰值）
- 其余 Linear：动态 W8A8 int8（torch._int_mm IMMA，权重 9 GB；m≤16 时 pad
  到 32）；embedding int8 行量化。两侧（TP1/TP2、两 rank）权重逐位一致，
  输出只差归约重关联
- GatedDeltaNet 走 transformers 5.18 + fla/causal-conv1d 原生 kernel

**延迟**（greedy，prompt 18 tok，纯 decode 计时，50 tok 均值，双 3080）：

| 配置 | tok/s | 峰值显存 |
|---|---|---|
| TP=1 | **10.19** | 17.4 GiB |
| TP=2 barlink PG（零拷贝） | 9.65 | ~10 GiB/卡 |
| TP=2 NCCL（SHM） | 9.58 | ~10 GiB/卡 |

每 token 128 次 10KB rowwise allreduce。**结论：eager 执行下 TP=2 无加速、
两后端无差异**——torch.profiler 实测每 token ~1200 个 kernel launch，
host 派发 + launch 间隙 ~60 ms，远超 GPU 计算（TP=1 50 ms，TP=2 每 rank
38 ms）；TP 只减半 GPU 侧工作，固定 per-op 成本（eager glue、int8 GEMV
固定开销、同步等待）不减。barlink 25 µs vs NCCL 49 µs 的 per-op 优势
（§7）在本负载仅占 3-6%。与 §9（Qwen3.8 int8，同架构同结论）互相印证：
**下一步必须是 CUDA graphs / 编译化执行**才能把 TP 收益和 barlink 延迟
优势兑现（graphs 化后 allreduce 占比升至主导，零拷贝路径直接受益）。
注：barlink 的 allreduce 协议 kernel 数多（6/op），graphs 化时 barlink
侧还需要把 seq 计数器做成可注入参数（当前烘焙在 kernel 参数里）。

**正确性**：TP=2 两 rank 输出逐 token 一致（True）；vs TP=1 前 46 token
完全一致、第 47 token greedy 翻转（int8 噪声下近邻并列），token match
94%（判据 ≥90% + rank 一致，`results/correctness_*.json`）。

**core 改动**：`k_flag_wait` 的 `__nanosleep` 退避（原 cap 1 ms）换成纯
自旋——flag 在本地内存，轮询不耗链路带宽，而 `__nanosleep` 会向上取整到
计时器 tick，把 decode-TP 的每次等待放大到数百 µs。PG 微基准不变
（10KB 24.9 µs），run_pg.sh / FLUX2 demo 全套回归通过。

运行：`bash demos/qwen_tp/run.sh [tp1|barlink|nccl]`（等卡窗口+重试；
worker 加载期间持有 15 GiB 显存占位防其它实验插队）。

## 10. vLLM 接入（barlink PG 作为 vLLM TP 通信后端）

**状态（2026-10-04）：`--enforce-eager` 全链路跑通**；CUDA graph 捕获仍被
invalidate，原因未定位（见下）。环境：独立 `.venv-vllm`（py3.14 + torch
2.14.0 + vllm 0.30.0，NJU 镜像）。

接入件（本仓库）：

- `vllm_barlink/` shim 包：注册 barlink backend；**惰性**包
  `init_process_group`/`is_backend_available`（启动时 import barlink_sm86
  会让 vllm 的 EngineCore 进程在 fork TP worker 之前初始化 CUDA —— bisect
  实证，惰性化后消失）；把 vllm 0.30 硬编码的 `"cpu:gloo,cuda:nccl"` 改写为
  `cuda:barlink`；默认 `VLLM_DISABLE_PYNCCL=1` 让 CudaCommunicator 的
  all_reduce/all_gather 走 torch-PG 回退（== barlink）。
  `install_pth.sh <venv>` 在 venv 装 `.pth` 实现每次 python 启动自动加载。
- `vllm_barlink/entry.py`：blrun 硬编码 `<repo>/.venv/bin/python`，但
  ambient cap 跨 execve 存活 —— 跳板用 vllm venv 的 python 重_exec 目标
  脚本，cap 一路传到 vllm spawn 的每个 worker（各 rank 自己的 init_peer
  后自行丢权）。
- 扩展双构建：pybind ABI 不跨 torch 版本，`barlink_sm86/_C.torch<XY>.so`
  按 torch 版本选加载（`_load()` 自动探测）；vllm venv 构建命令：
  `cd torch_ext && rm -rf build-vllm && TORCH_CUDA_ARCH_LIST=8.6
  ../.venv-vllm/bin/python setup.py build_ext --inplace --build-lib build-vllm
  && cp build-vllm/barlink_sm86/_C*.so barlink_sm86/_C.torch214.so`
  （**注意 torch 2.13.0 的 `Backend` pybind 没有构造器** ——
  "No constructor defined!"，python 扩展后端在 2.13.0 上完全无法实例化，
  2.14 已修；vllm 0.27–0.30 全部钉死 torch==2.13.0，故 vllm venv 用
  torch 2.14.0 + torchvision 0.29.0 + triton 3.8.0 --no-deps 覆盖，实测
  vllm 0.30 可正常运行）
- PG 层新能力（全部带测试）：`all_gather_single`/`allgather`（对端半片经
  p2p 零拷贝全双工交换）、size-1 子组的 `_DummyBackend1` 诚实本地语义
  （vllm 会建 singleton 子组，旧实现 raise 直接崩初始化；dummy 不触碰链路
  单例）。

```bash
# 冒烟（TP=2, Qwen2-1.5B, graph 关闭）：
PATH="$PWD/.venv-vllm/bin:$PATH" ENFORCE_EAGER=1 BL_SKIP_INIT=1 BL_POOL_MB=192 \
    HF_HUB_OFFLINE=1 tools/blrun vllm_barlink/entry.py demos/vllm_bl/smoke_vllm.py
# 输出示例：' John and I am a 2017 graduate of the University of Texas at ...'
```

**两个平台级发现（重要）**：

1. **BAR 写入完成（ack） pace ~7–10 MB/s，交付（delivery）13 GB/s**。
   `__threadfence_system`/kernel 完成会等全部 outstanding BAR store 的完成
   确认，对端有读/轮询负载时该确认以 ~7-10 MB/s 推进（16MB payload ≈
   2.4s）。§6 的 13.2 GB/s 是 event 口径的交付带宽，不含完成等待。后果：
   大消息（>1MB）的 op 延迟是秒级；已从 `k_mark` 移除 fence（冗余且有害：
   kernel 边界 + 单源 PCIe 保序已足够，全回归通过）；所有 flag 等待超时
   提到 ~200e9 时钟。小消息（≤64KB，LLM decode 场景）不受影响（10KB
   allreduce 仍 ~19µs）。**多 chunk 全双工 p2p（>32MB 分片）有一个未解的
   flag 送达病理（~4.4s 上下文死亡，coredump 显示 k_flag_wait SIGTRAP 但
   时序对不上任何超时预算）** —— PG 层 p2p 分片因此上限提到整区（47MB，
   单 chunk），大消息走单 chunk（慢但正确）。
2. **CUDA graph 捕获**：隔离复现（裸 CUDAGraph + dist.all_reduce，fused 和
   旧路径、单次/重复 replay）**全部通过**；但在 vllm piecewise 捕获里
   allreduce 首次进入图时 capture invalidated（`joinSendStream` 的跨流
   event 等待已做 capture 跳过，仍失败；vllm worker 会剥离自定义环境变量，
   埋点需要在 core 里硬编码）。`--enforce-eager` 跑通在先，捕获问题留作
   TODO（方向：piecewise 捕获下 c10d 扩展后端的交互，或给 vllm 的
   all_reduce op 注册为 splitting op 使其不进图）。

### 27B INT8 decode TPS 实测（2026-10-04，干净卡）

vLLM 0.30 原生支持 Qwen3_5 架构（`Qwen3_5ForConditionalGeneration` 在
registry；GDN 层走 Triton prefill + CUDA decode kernel；compressed-tensors
W8A16 直接加载）。`--enforce-eager`（graph 捕获仍受阻，见上）：

| backend | tok/s（batch 4 × 256 tok） | wall |
|---|---|---|
| barlink PG（enforce-eager） | 62.7 / 61.8（均值 ~62.2） | 16.3–16.6 s |
| NCCL（eager） | 65.4 / 64.5（均值 ~64.9） | 15.7–15.9 s |
| **barlink PG（CUDA graph）** | **150.0 / 149.9** | 6.8 s |
| NCCL（CUDA graph） | 143.9 / 144.6（均值 ~144.2） | 7.1 s |

graph 模式（FULL_AND_PIECEWISE，replay-safe 协议后不再需要
--enforce-eager）：barlink 比 eager 快 **2.4×**，且反超 NCCL **~4%** ——
eager 的每 op Python→c10d→PG 派发开销消失后，fused 协议的链路延迟优势
显现。

- 两 backend greedy 输出逐字一致（同一 int8 kernel + 等价加法语义）。
- NCCL 快 ~4%：enforce-eager 下每 op 走 Python→c10d→PG 派发，barlink 的
  6/2-kernel 协议比 NCCL 单 kernel 多一点固定开销；差距比在自定义 decode
  路径上的 ~1-2% 略大但仍同量级。
- 对照自定义 decode 路径（§9，batch 1 手写循环）：barlink 8.11 / NCCL 8.01
  tok/s —— vLLM 快 ~7.7×（fused kernel + 批量 + 成熟 GEMM），通信后端
  差异在两种 harness 下都只有几个百分点。
- 复现：`demos/vllm_bl/bench_27b.py`（barlink 经 entry.py 跳板；NCCL 加
  `BL_SHIM_OFF=1` 直跑 venv python）；结果 json 在同目录。

为跑通 27B 补的三个 PG/core 修复（均有回归）：① 大 allgather（vllm
profile_run 的 padded logits gather，>p2p 区 48MB）改走 **pool 暂存 +
bl.copy_ 臂路径交换 + 新 `bl.pool_move`（k_move 语义显式源）**，绕开多
chunk 全双工 p2p 的未解病理；暂存用一对 16MB 复用 buffer（per-size 缓存
会耗尽 pool）。② PG `barrier` 改零拷贝 fused allreduce（pool allreduce_
要求 tensor 落在 ar scratch 镜像区内，大暂存分配后 bump 越界）。③ 修复
core.cu 提交态不一致（BL_CAPDBG 定义在使用之后、initWaitCap 孤儿调用、
NO_SIDESTREAM 诊断块引用未声明变量 —— 历史多轮编辑漂移）。
