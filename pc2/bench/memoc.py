# Usage: memoc.py            -> show offsets
#        memoc.py <MHz> [idx...] -> set memory VF offset (NVML units) on given GPUs (default all)
import sys
from pynvml import *
nvmlInit()
idx = [int(a) for a in sys.argv[2:]] or range(nvmlDeviceGetCount())
for i in idx:
    h = nvmlDeviceGetHandleByIndex(i)
    if len(sys.argv) > 1:
        nvmlDeviceSetMemClkVfOffset(h, int(sys.argv[1]))
    lo, hi = nvmlDeviceGetMemClkMinMaxVfOffset(h)
    print(f"GPU{i} mem offset {nvmlDeviceGetMemClkVfOffset(h)} (range {lo}..{hi}), "
          f"mem clock now {nvmlDeviceGetClockInfo(h, NVML_CLOCK_MEM)} MHz, max {nvmlDeviceGetMaxClockInfo(h, NVML_CLOCK_MEM)} MHz")
