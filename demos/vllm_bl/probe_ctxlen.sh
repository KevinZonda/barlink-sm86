#!/bin/bash
cd /home/kevin/projects/nv-p2p/barlink-torch
for LEN in 32768 65536 98304 131072 196608 262144; do
  echo "##### probing max_model_len=$LEN $(date)"
  LOG=demos/vllm_bl/probe_ctx_${LEN}.log
  PATH="$PWD/.venv-vllm/bin:$PATH" BL_SHIM_OFF=1 HF_HUB_OFFLINE=1 NCCL_P2P_DISABLE=1 \
    timeout 600 .venv-vllm/bin/python demos/vllm_bl/bench_27b.py \
    --mtp 3 --max-num-seqs 1 --gpu-mem 0.93 \
    --prompt-tokens 512 --max-tokens 8 --max-model-len $LEN \
    --kv-dtype fp8_e4m3 --chunked-prefill \
    --out-json demos/vllm_bl/probe_ctx_${LEN}.json > $LOG 2>&1
  RC=$?
  KVTOK=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" $LOG | tail -1)
  MAMBA=$(grep -oE "Mamba cache blocks: [0-9]+|num_mamba_blocks=[0-9]+|[0-9]+ mamba blocks" $LOG | tail -1)
  echo "##### LEN=$LEN rc=$RC $KVTOK $MAMBA"
  [ $RC -ne 0 ] && grep -iE "out of memory|no available memory|exceeds|error" $LOG | head -3
done
echo "##### sweep done $(date)"
