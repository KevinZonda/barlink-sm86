# Qwen3.8-27B INT8 W8A16 TP=2 decode：barlink PG vs NCCL（2026-10-04）

模型：`/mnt/modelzoo/lued/Qwen3.8-27B-INT8-W8A16-MTP`（compressed-tensors
pack-quantized INT8 W8A16，group 128，symmetric；64 层 = 48 GDN 线性注意力 + 16
全注意力；hidden 5120，vocab 248320）。双 RTX 3080 20GB，transformers 5.18.0，
torch 2.14.0+cu130。

## 结果（greedy decode，prompt 42 tok，warmup 16，计时 64 tok，两 rank 各自计时）

| backend | tok/s (rank0/rank1) | decode ms/token | 通信开销 ms/token* |
|---|---|---|---|
| barlink PG | 6.96（三次运行 6.63 / 6.96 / 7.27） | 132 | **32.0** |
| NCCL（SHM 中转） | **8.03**（8.02 / 8.03） | 121 | 12.5 |
| barlink 无通信对照 | 8.95 | 110 | — |
| NCCL 无通信对照 | 8.93 | 110 | — |

* 通信开销 = 1/tok_s_full − 1/tok_s_nocomm。每 token 通信 = 128 次 rowwise
all_reduce（每 down/o/out_proj 一次，10KB bf16）+ 1 次 lm_head 半 logits 交换
（248KB）。摊到单次 all_reduce：barlink ~250 µs，NCCL ~98 µs。

**结论**：这个负载下 NCCL 快 ~13-15%。原因不是链路延迟（§7 微基准：barlink
10KB allreduce 25 µs < NCCL-SHM 49 µs），而是**每 op 的 kernel 数**：barlink
allreduce 协议是 6 个 stream-ordered kernel（arm/armwait/payload/mark/
flagwait/add），decode 是串行关键路径（每 token ~4500 个小 kernel，GPU 大量
空转等 launch），每 op 多出的 kernel launch 全部落在关键路径上；NCCL allreduce
单 kernel 完成。GDN 走 transformers 原生 torch 慢速 kernel（无 fla），进一步
放大了 launch-bound 程度（每 token 仅 ~11.3GB 权重读取，110ms → 103GB/s，
远低于显存带宽上限）。

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

## 复现

```bash
bash demos/qwen_tp/qwen38_int8/run.sh all   # barlink + nccl + 无通信对照 + 比对
```
