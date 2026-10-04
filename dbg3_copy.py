import os
import barlink_sm86 as bl

print("cwd", os.getcwd(), flush=True)
print("pre :", open("torch_ext/tests/test_basic.py", "rb").read(28), flush=True)
bl.init(devices=[0, 1], pool_mb=64)
fd = os.open("torch_ext/tests/test_basic.py", os.O_RDONLY)
d = os.read(fd, 64)
print("post:", d[:28], "nulls_in_64:", d.count(b"\x00"), flush=True)
print("stat size:", os.stat("torch_ext/tests/test_basic.py").st_size, flush=True)
print("other file:", open("tools/blrun.c", "rb").read(20), flush=True)
