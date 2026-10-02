# Payload-side security assertion: by the time user code runs, the process
# must have zero capabilities and no_new_privs set. Exits 1 if any cap leaks.
import sys

import barlink_sm86 as bl  # already initialized by blrun's bootstrap

status = {}
for line in open("/proc/self/status"):
    parts = line.split()
    if len(parts) >= 2:
        status[parts[0].rstrip(":")] = parts[1]

capeff = int(status["CapEff"], 16)
capamb = int(status["CapAmb"], 16)
capprm = int(status["CapPrm"], 16)
nnp = int(status["NoNewPrivs"])

print("payload view: CapEff=%#x CapPrm=%#x CapAmb=%#x NoNewPrivs=%d"
      % (capeff, capprm, capamb, nnp))

assert capeff == 0 and capamb == 0, "capability leak into payload!"
assert nnp == 1, "no_new_privs lost!"

# pool is live: a real copy_ round-trip works from capless user code
import torch
a = bl.empty(1 << 20, device=0)
b = bl.empty(1 << 20, device=1)
a.random_(0, 256)
bl.copy_(b, a)
assert bl.verify() == 0
print("payload copy_ round-trip OK -- pool usable with zero caps")
