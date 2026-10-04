import os
p = "torch_ext/tests/repro_p2p_big.py"
bad = 0
for i in range(2000):
    d = open(p, "rb").read()
    if b"\x00" in d or len(d) != 2945:
        bad += 1
        if bad < 4:
            print("iter", i, "len", len(d), "nulls", d.count(b"\x00"), flush=True)
print("bad reads:", bad, "/2000", flush=True)
