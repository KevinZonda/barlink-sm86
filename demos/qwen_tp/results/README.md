# Qwen3.5-27B-GPTQ-Int4 TP decode results（2026-10-04）

模型：/mnt/modelzoo/Qwen/Qwen3.5-27B-GPTQ-Int4（hf-mirror/ModelScope，text-only
加载：丢 vision tower + MTP head）。固定 prompt（18 token，"Explain how GPU P2P
communication over PCIe BAR1..."），greedy，batch 1，纯 decode 计时（prefill +
warmup 不计入），50 token 均值。每步 128 次 rowwise allreduce（o_proj /
linear_attn.out_proj / mlp.down_proj 各层后，10 KB bf16）。

| 配置 | tok/s | 峰值显存 |
|---|---|---|
| TP=1（单卡） | **10.19** | 17.4 GiB |
| TP=2 barlink PG（零拷贝） | **9.65** | ~10 GiB/卡 |
| TP=2 NCCL（SHM，无 P2P） | **9.58** | ~10 GiB/卡 |

正确性：TP=2 两 rank 输出逐 token 一致；vs TP=1 前 46 token 完全一致，
第 47 token greedy 翻转（int8 reassociation 噪声下的近邻并列），
token match 94%（compare.py 判据 ≥90% + rank 一致）。两后端数字一致。

**结论**：本 eager 实现下 TP=2 相对 TP=1 无加速（9.6 vs 10.2 tok/s），
barlink 与 NCCL 无差异。原因（torch.profiler 实测，单 token）：
~1200 个 kernel launch/token，host 派发 + launch 间隙 ~60 ms/token，
远大于 GPU 计算时间（TP=1 50 ms，TP=2 每 rank 38 ms）。TP 只减半了
GPU 侧工作，固定 per-op 成本（eager glue、int8 GEMV 固定开销、allreduce
同步）不减。barlink 零拷贝 vs NCCL 的 per-op 优势（25 µs vs 49 µs @10KB，
§7）在本负载仅占 ~3-6%，被派发开销淹没。下一步必须是 CUDA graphs /
编译化执行（vLLM 类引擎）才能释放 TP 收益——barlink 的延迟优势在那种
执行模型下才会兑现（ graphs 化后 allreduce 占比会升至主导）。

权重处理（worker.py 头注释有完整说明）：checkpoint 只有 MLP 是 int4
（其余 bf16，单卡放不下）→ MLP 走自写 Triton int4 GEMV（544 GB/s eff，
v1 布局直读、TP 切分为 group 对齐的精确子集）；其余 Linear 动态 W8A8
int8（torch._int_mm，权重 9 GB）；embedding int8。

文件：stats_tp{1,2,2_nccl}.json、genids_*.pt、correctness_*.json、
tp1.log、tp{,2_nccl}_rank{0,1}.log。重跑：
`bash demos/qwen_tp/run.sh [tp1|barlink|nccl]`。
