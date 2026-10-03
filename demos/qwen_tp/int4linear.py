# SPDX-License-Identifier: MIT
#
# Standalone int4 weight-only linear for the Qwen TP decode demo.
#
# Layout is the GPTQ v1 convention (same as the checkpoint):
#   qweight int32 [IN/8, OUT]   -- 8 consecutive k per int32, nibble = k % 8
#   scales  fp16  [groups, OUT] -- one fp16 scale per (128-input group, out)
#   qzeros  int32 [groups, OUT/8] -- 8 consecutive out per int32, nibble = n%8
# Quantization rule (symmetric): dequant = (nibble - zero) * scale, zero=8.
#
# Decode (bs*seq == 1) runs a small Triton GEMV that reads exactly the int4
# bytes once; longer sequences take a torch unpack path (prefill only, not
# part of the decode timing).

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gemv_int4_kernel(x_ptr, qw_ptr, sc_ptr, qz_ptr, y_ptr,
                      IN, OUT,
                      GS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_off = pid * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], dtype=tl.float32)
    kin = IN // tl.num_programs(1)
    k_lo = pid_k * kin
    for k0 in range(k_lo, k_lo + kin, BK):
        g = k0 // GS
        sc = tl.load(sc_ptr + g * OUT + n_off).to(tl.float32)          # [BN]
        qzv = tl.load(qz_ptr + g * (OUT // 8) + n_off // 8)            # [BN]
        zp = ((qzv >> ((n_off % 8) * 4)) & 0xF).to(tl.float32)
        ko = k0 // 8 + tl.arange(0, BK // 8)
        pk = tl.load(qw_ptr + ko[:, None] * OUT + n_off[None, :])      # [BK/8, BN]
        io = tl.arange(0, BK // 8)
        for j in tl.static_range(8):
            nib = ((pk >> (j * 4)) & 0xF).to(tl.float32)
            xj = tl.load(x_ptr + k0 + j + 8 * io).to(tl.float32)       # [BK/8]
            acc += tl.sum(xj[:, None] * (nib - zp[None, :]) * sc[None, :], 0)
    tl.atomic_add(y_ptr + n_off, acc)


def pack_int4(weight_bf16: torch.Tensor, group_size: int = 128):
    """[OUT, IN] fp/bf16 -> (qweight, scales, qzeros) v1 layout, sym zero=8."""
    w = weight_bf16.float()
    out_f, in_f = w.shape
    assert in_f % group_size == 0
    groups = in_f // group_size
    wg = w.reshape(out_f, groups, group_size)
    scale = (wg.abs().amax(dim=2) / 7.0).clamp(min=1e-8)          # [OUT, groups]
    q = torch.round(wg / scale[..., None] + 8.0).clamp(0, 15).to(torch.int32)
    q = q.reshape(out_f, in_f)
    scale = scale.t().contiguous().half()                          # [groups, OUT]
    # pack along k: qweight [IN/8, OUT]
    qk = q.reshape(out_f, in_f // 8, 8)
    qw = (qk[..., 0] | (qk[..., 1] << 4) | (qk[..., 2] << 8)
          | (qk[..., 3] << 12) | (qk[..., 4] << 16) | (qk[..., 5] << 20)
          | (qk[..., 6] << 24) | (qk[..., 7] << 28))
    qw = qw.t().contiguous().to(torch.int32)                       # [IN/8, OUT]
    zeros = torch.full((groups, out_f), 8, dtype=torch.int32)
    qz = zeros.reshape(groups, out_f // 8, 8)
    qz = (qz[..., 0] | (qz[..., 1] << 4) | (qz[..., 2] << 8)
          | (qz[..., 3] << 12) | (qz[..., 4] << 16) | (qz[..., 5] << 20)
          | (qz[..., 6] << 24) | (qz[..., 7] << 28)).contiguous()
    return qw, scale, qz


class Int4Linear(nn.Module):
    def __init__(self, weight_bf16, group_size=128):
        super().__init__()
        self.in_features, self.out_features = weight_bf16.shape[1], weight_bf16.shape[0]
        self.group_size = group_size
        qw, sc, qz = pack_int4(weight_bf16, group_size)
        self.register_buffer("qweight", qw)
        self.register_buffer("scales", sc)
        self.register_buffer("qzeros", qz)

    @classmethod
    def from_gptq_v1(cls, qweight, scales, qzeros, g_idx, group_size=128):
        """Build directly from checkpoint GPTQ v1 tensors (no requant):
        qweight [IN/8, OUT] int32, scales [groups, OUT], qzeros [groups, OUT/8],
        g_idx [IN]. Requires desc_act=False (identity groups)."""
        in_f = qweight.shape[0] * 8
        out_f = qweight.shape[1]
        assert scales.shape == (in_f // group_size, out_f), scales.shape
        assert qzeros.shape == (in_f // group_size, out_f // 8), qzeros.shape
        g = g_idx.reshape(-1, group_size)
        assert bool((g == g[:, :1]).all()) and \
            bool((g[1:, 0] == g[:-1, 0] + 1).all()), \
            "desc_act/group-misaligned g_idx not supported"
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.in_features, m.out_features, m.group_size = in_f, out_f, group_size
        m.register_buffer("qweight", qweight.contiguous())
        m.register_buffer("scales", scales.contiguous())
        m.register_buffer("qzeros", qzeros.contiguous())
        return m

    def _dequant(self):
        w = self.qweight.t().long()                                   # [OUT, IN/8]
        nib = torch.stack([(w >> (4 * j)) & 0xF for j in range(8)], dim=2)
        nib = nib.reshape(self.out_features, self.in_features)
        z = torch.stack([(self.qzeros.long() >> (4 * j)) & 0xF
                         for j in range(8)], dim=2)
        z = z.reshape(-1, self.out_features)                          # [IN, OUT]
        z = z.repeat_interleave(self.group_size, dim=0)[:self.in_features].t()
        sc = self.scales.repeat_interleave(self.group_size, dim=0)[:self.in_features].t()
        return ((nib - z) * sc.float()).to(torch.bfloat16)

    def forward(self, x):
        shape = x.shape[:-1]
        if x.numel() // x.shape[-1] == 1:
            x2 = x.reshape(-1).to(torch.bfloat16).contiguous()
            y = torch.zeros(self.out_features, dtype=torch.float32,
                            device=x.device)
            BN, BK, SPLIT = 128, 256, 2
            grid = (self.out_features // BN, SPLIT)
            assert self.in_features % (SPLIT * self.group_size) == 0
            assert self.in_features % BK == 0 and self.out_features % BN == 0
            _gemv_int4_kernel[grid](x2, self.qweight, self.scales,
                                    self.qzeros, y,
                                    self.in_features, self.out_features,
                                    GS=self.group_size, BN=BN, BK=BK,
                                    num_warps=4)
            return y.to(torch.bfloat16).reshape(*shape, self.out_features)
        # prefill path: chunked unpack + matmul (not decode-timed)
        x2 = x.reshape(-1, self.in_features)
        outs = []
        step = max(1, (16 << 20) // max(1, self.in_features))
        for o0 in range(0, self.out_features, step):
            o1 = min(o0 + step, self.out_features)
            outs.append(x2 @ self._dequant_slice(o0, o1).t())
        return torch.cat(outs, dim=1).reshape(*shape, self.out_features)

    def _dequant_slice(self, o0, o1):
        w = self.qweight.t()[o0:o1].long()
        nib = torch.stack([(w >> (4 * j)) & 0xF for j in range(8)], dim=2)
        nib = nib.reshape(o1 - o0, self.in_features)
        zfull = torch.stack([(self.qzeros.long() >> (4 * j)) & 0xF
                             for j in range(8)], dim=2)
        zfull = zfull.reshape(-1, self.out_features)[:, o0:o1]        # [IN, o]
        z = zfull.repeat_interleave(self.group_size, dim=0)[:self.in_features].t()
        sc = self.scales.repeat_interleave(self.group_size, dim=0)[:self.in_features, o0:o1].t()
        return ((nib - z) * sc.float()).to(torch.bfloat16)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    for (i, o) in ((5120, 17408), (17408, 5120), (5120, 10240), (5120, 6144)):
        w = torch.randn(o, i, dtype=torch.bfloat16) * 0.02
        m = Int4Linear(w).cuda()
        x = torch.randn(1, i, dtype=torch.bfloat16, device="cuda")
        y = m(x)
        dq = m._dequant()
        yr = (x.float() @ dq.t().float().cuda())
        err = (y.float() - yr).abs().max().item()
        rel = err / yr.abs().max().item()
        for _ in range(10):
            m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            m(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 200
        print("[%dx%d] rel_err=%.5f gemv=%.4f ms -> %.0f GB/s eff"
              % (i, o, rel, dt * 1e3, i * o / 2 / dt / 1e9))
