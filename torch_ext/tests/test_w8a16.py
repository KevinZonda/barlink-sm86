import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import barlink_sm86 as bl

torch.manual_seed(7)
dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)

def prep(N, K, dev):
    G = K // 128
    wq = torch.randint(-128, 128, (N, K), dtype=torch.int8, device=dev)
    sw = (torch.rand(N, G, dtype=torch.float32, device=dev) * 0.01 + 0.0001)
    swb = sw.bfloat16()          # checkpoint stores bf16 scales
    # transposed layouts (what the worker will build at load)
    wq_t = wq.view(N, K // 16, 16).permute(1, 0, 2).contiguous()
    sw_t = swb.float().t().contiguous()     # [G, N] fp32
    return wq, swb, wq_t, sw_t

def ref_fp64(x, wq, swb):
    w = wq.double() * swb.double().repeat_interleave(128, dim=1)
    return x.double() @ w.t()

shapes = [  # (N, K) decode-relevant, TP=2 sharded
    (8704, 5120),   # gate/up
    (5120, 8704),   # down
    (6144, 5120),   # q_proj full-attn
    (512, 5120),    # k/v
    (5120, 3072),   # o / gdn out
    (5120, 5120),   # gdn in_proj_qkv
    (3072, 5120),   # gdn z
    (124160, 5120), # lm_head half vocab
]
ok_all = True
for N, K in shapes:
    wq, swb, wq_t, sw_t = prep(N, K, dev)
    for M in (1, 3, 16):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 2
        xq, xs = bl._C.w8a16_quant_act(x)
        # quant reference (fp64 of the quantized values)
        xdq = xq.double() * xs.double()[:, None]
        ref = ref_fp64(xdq, wq, swb)
        for S in (1, 4, 7):
            if S > K // 128: continue
            y = bl._C.w8a16_gemm(xq, xs, wq_t, sw_t, N, K, S)
            d = (y.double() - ref)
            rel = (d.norm() / ref.norm()).item()
            mx = d.abs().max().item()
            status = "OK" if rel < 2e-2 else "FAIL"
            if rel >= 2e-2: ok_all = False
            print("N=%6d K=%5d M=%2d S=%d rel_l2=%.6f max_abs=%.4f %s"
                  % (N, K, M, S, rel, mx, status))
print("ALL OK" if ok_all else "FAILURES")
