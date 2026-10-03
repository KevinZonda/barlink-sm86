# FLUX.2 Klein 9B TP demo results（2026-10-03）

模型：/mnt/modelzoo/black-forest-labs/FLUX.2-klein-base-9B/transformer（bf16）
输入：img 4096 token（64×64 latent）+ text 512×12288，seed 1234，batch 1，
inference_mode forward，CUDA event 计时，10 步（3 步 warmup）。

| 配置 | ms/step mean (min–max) | vs TP=1 |
|---|---|---|
| TP=1 | 1393.3 (1384.6–1401.7) | 1.00× |
| TP=2 barlink PG（零拷贝） | 896.4 (891.5–905.7) | **1.55×** |
| TP=2 NCCL（SHM，无 P2P） | 918.2 (916.0–922.1) | 1.52× |
| TP=2 gloo（host staging） | 1564.1 | 0.89× |

barlink vs NCCL：step 时间快 2.4%（此负载带宽主导）；小消息固定延迟差距大
（PG 10KB：25µs vs NCCL-SHM 49µs，见 BUILD_AND_TEST.md §7）。gloo 比单卡慢。
三后端正确性一致：cosine 0.9998，rel_L2 1.77%（bf16 噪声底内，anchor 论证见
§8 / correctness*.json）。

文件：
- `stats_tp{1,2}.json` / `stats_tp2{nccl,gloo}.json` / `stats_tp2anchor.json`
- `output_*.pt`：rank0 输出（fp32 CPU）
- `correctness.json`（barlink）/ `correctness_tp2{nccl,gloo}.json`
- `tp1.log` / `tp2_rank{0,1}.log` / `tp2{nccl,gloo}_rank{0,1}.log`

重跑：`bash demos/flux2_tp/run.sh [barlink|nccl|gloo]`（自动等卡、重试）。
