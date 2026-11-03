import sys
from pynvml import *
nvmlInit()
vals = [int(a) for a in sys.argv[1:]]
for i in range(nvmlDeviceGetCount()):
    h = nvmlDeviceGetHandleByIndex(i)
    if vals:
        nvmlDeviceSetGpcClkVfOffset(h, vals[i])
    print(f"GPU{i} gpc offset {nvmlDeviceGetGpcClkVfOffset(h)} mem offset {nvmlDeviceGetMemClkVfOffset(h)}")
