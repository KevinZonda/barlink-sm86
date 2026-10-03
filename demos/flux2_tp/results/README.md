# FLUX.2 Klein 9B TP demo results（2026-10-03）

模型：/mnt/modelzoo/black-forest-labs/FLUX.2-klein-base-9B/transformer（bf16）
输入：img 4096 token（64×64 latent）+ text 512×12288，seed 1234，batch 1，
inference_mode forward，CUDA event 计时，10 步（3 步 warmup）。

| 配置 | ms/step mean (min–max) | 峰值显存 |
|---|---|---|
| TP=1 | 1401.3 (1391.9–1410.3) | 17.7 GiB |
| TP=2 (barlink PG) | 927.1 (922.3–929.7) | 16.9 GiB |

加速比 1.51×。正确性：cosine 0.9998，rel_L2 1.77%，anchor（fp32 计算）噪声
底 TP1 2.31% / TP2 2.42% → PASS。详见 BUILD_AND_TEST.md §8 与
correctness.json。

文件：
- `stats_tp{1,2}.json` / `stats_tp2anchor.json`：每步时间与配置
- `output_tp{1,2}.pt` / `output_tp2anchor.pt`：rank0 输出（fp32 CPU）
- `correctness.json` / `correctness.log`：比对结果
- `tp1.log` / `tp2_rank{0,1}.log`：运行日志（两进程各一份）

重跑：`bash demos/flux2_tp/run.sh`（自动等卡、重试）。
