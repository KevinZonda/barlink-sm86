# Qwen3.8-27B INT8 W8A16 TP=2 decode：barlink PG vs NCCL（2026-10-04）

模型：`/mnt/modelzoo/lued/Qwen3.8-27B-INT8-W8A16-MTP`（compressed-tensors
pack-quantized INT8 W8A16，group 128，symmetric；64 层 = 48 GDN 线性注意力 + 16
全注意力；hidden 5120，vocab 248320）。双 RTX 3080 20GB，transformers 5.18.0，
torch 2.14.0+cu130。

## 结果（greedy decode，prompt 42 tok，warmup 16，计时 64 tok，两 rank 各自计时）

| backend | tok/s (rank0/rank1) | decode ms/token | 通信开销 ms/token* |
|---|---|---|---|
| barlink PG | ~~6.96~~ **8.03**（干净卡；早期 6.63–7.27 为共卡污染） | 122 | **12.8** |
| NCCL（SHM 中转） | **8.03**（8.02 / 8.03） | 121 | 12.5 |
| barlink 无通信对照 | 8.95 | 110 | — |
| NCCL 无通信对照 | 8.93 | 110 | — |

* 通信开销 = 1/tok_s_full − 1/tok_s_nocomm。每 token 通信 = 128 次 rowwise
all_reduce（每 down/o/out_proj 一次，10KB bf16）+ 1 次 lm_head 半 logits 交换
（248KB）。摊到单次 all_reduce：两 backend 都是 ~95–105 µs。

**结论（已按干净卡修正）**：两 backend **平手**（12.8 vs 12.5 ms/token 通信
开销，远低于此前的误判 32 vs 12.5——当时另一会话在卡上轮流跑任务，其
PCIe/IOMMU 流量不对称放大了 barlink 跨卡 flag 轮询延迟，见下文 with-fla
一节）。通信开销占 decode（122ms）约 10%；其余 ~110ms 是无通信对照也存在的
固定成本（见下方 with-fla 一节的 profiler 分解：~29k 小 kernel/token，int8
gemm 只占 21ms）。

## 正确性（compare.py）

- 两 backend rank 各自一致；**barlink 与 NCCL 输出逐位相同**（64 token 全
  match，首 token logits rel_l2 = 0.0）——两 rank 加法域内 fp32 求和一次舍入，
  与 NCCL bf16 SUM（fp32 累加）对 2-rank 情形逐位一致。
- `correctness_barlink_vs_nccl.json`。

## 权重与计算路径（两 backend 完全一致）

- 两 rank 各自流式完整加载 checkpoint，本地切 shard（无 c10d scatter）。
- pack-quantized int32 解包约定：**bias-128 存储**（`packed.view(int8) ^ -128`，
  与 compressed_tensors unpacker 逐元素验证）。group-128 scale 在 fp32 下精确
  fold 成 per-row int8（第二次量化，两侧一致，不影响对比）。
- matmul = 动态 per-token 激活 int8 量化 + `torch._int_mm`（cublasLt IMMA，
  sm80+；**m≤16 时 pad 到 32**，cublasLt 要求 m>16）。decode 读取 1 B/param。
- lm_head（2.5GB bf16）colwise 切 vocab + 半 logits 交换拼接：barlink 走
  isend/irecv 零拷贝 p2p；**NCCL 走 all_gather** —— torch 2.14 的 NCCL lazy
  p2p communicator 在本平台裸 isend/irecv 即 illegal memory access（最小复现
  双卡必现，与 NCCL_P2P_DISABLE 无关）。
- 每卡峰值显存 14.9 GiB（int8 shard 11.3 + embed 2.4 + lm_head 1.2 + 杂项）。

## fused 小消息协议（2026-10-04）

allreduce payload ≤64KB 走 2-kernel fused 路径（consumed-receipt 门控 + 本地
原子 last-block 选举，专用 scratch 分区），>64KB 保持 6-kernel 旧路径。
隔离口径 10KB allreduce 24.4→16.4 µs（`bench/latency/trials/fused-ar/`）；
本 bench（tag `_fused`）：barlink **8.11** vs NCCL 8.01 tok/s，barlink 通信
开销 12.8→**10.0 ms/token**，正确性仍逐位一致。BAR 原子（red/atom.global
对 peer BAR1）功能可用但毫秒级延迟，弃用（见 BUILD_AND_TEST.md §7）。

## 复现

```bash
bash demos/qwen_tp/qwen38_int8/run.sh all   # barlink + nccl + 无通信对照 + 比对
# fla/causal-conv1d 优化 kernel 环境下另存一套 tag，不覆盖旧结果：
QWEN38_TAG_SUFFIX=_fla bash demos/qwen_tp/qwen38_int8/run.sh all
```

## with fla kernels（2026-10-04 补测）

装上 `causal-conv1d 1.7.0`（源码编译 CUDA 扩展，另一会话编译，sm86 .so
265MB）和 `flash-linear-attention 0.5.2`（纯 python/triton）后，transformers
的 hub-kernel 解析自动走优化路径（日志里 "falling back to reference PyTorch
implementation" 警告消失）。**正确性仍然逐位一致**：fla 与原生路径逐位相同
（step-0 logits rel_l2 = 0.0），barlink 与 NCCL 也逐位相同。

| 环境 | barlink PG | NCCL | 无通信对照 | barlink 通信开销 |
|---|---|---|---|---|
| 原生 torch GDN（干净卡） | **8.03** | 8.03 | 8.95 / 8.93 | 12.8 ms/token |
| fla 优化 GDN | 7.95（7.82–7.98） | 8.11（8.02–8.11） | 8.80 / 8.97 | 13.2 ms/token |

**重要修正**：第一版 §9 的"barlink 6.96 vs NCCL 8.03、NCCL 快 15%"是**共卡
污染假象**——当时另一会话在两张卡上轮流跑 15.4GB 任务（日志里大片 OOM 重试
为证），其 PCIe/IOMMU 流量与 GPU 争抢不对称地放大了 barlink 跨卡 flag 轮询的
延迟。**干净卡下两种环境都是平手**（~12-13 ms/token 通信开销，摊到 128 次
10KB all_reduce ≈ 95-105 µs/次）。

**fla 为什么没有提速**（profiler 实测，`profile_decode.py`）：每 token 28.7k
个 CUDA kernel、96ms device 时间，其中 400 个 `_int_mm` 只有 21ms——瓶颈是
~24k 个小 elementwise/copy kernel（动态量化 wrapper 每次 ~10 个 + HF glue），
GDN 原生路径只占 ~3.8k 个（fla 砍掉后 28.7k vs 32.5k，省 ~5ms，被噪声淹没）。
fla 环境 barlink 与 NCCL 依旧平手说明：通信开销里**链路延迟和协议 kernel 数
都不是当前矛盾**（~100µs/次 vs §7 微基准 25µs，差距来自串行关键路径上的
enqueue/等待交叠），真正的大头是模型本身的 kernel 密度。

**对 fused 小消息协议优先级的启示**：不迫切。通信开销 12-13ms/token 里
barlink 与 NCCL 无差，压 barlink 协议 kernel 数（6→3）最多再省几 ms；而
动态量化 wrapper 的 ~4000 个 kernel（可 fuse 成 per-layer 单 kernel 或换
真 W8A16 kernel）和 HF glue 才是 ~70ms 的富矿。若做 LLM 场景优先级：
weight-only int8 kernel（Marlin 类）> 量化/反量化 fuse > 小消息协议 fuse。
