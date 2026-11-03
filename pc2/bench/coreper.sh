#!/bin/bash
# Per-card core offsets aimed at equal clocks (TP steps wait for the slowest rank). Each level runs about 4.5 min.
# Stops on Xid, container restart, corrupt output, failed check or hot GPU; always ends at core offset 0.
cd ~/bench
cat > gpuoc.py <<'E'
import sys
from pynvml import *
nvmlInit()
vals = [int(a) for a in sys.argv[1:]]
for i in range(nvmlDeviceGetCount()):
    h = nvmlDeviceGetHandleByIndex(i)
    if vals:
        nvmlDeviceSetGpcClkVfOffset(h, vals[i])
    print(f"GPU{i} gpc offset {nvmlDeviceGetGpcClkVfOffset(h)} mem offset {nvmlDeviceGetMemClkVfOffset(h)}")
E
OC="sudo /opt/gpu-oc/bin/python gpuoc.py"
finish(){ echo "== reset core offset 0 ($(date +%T))"; $OC 0 0 0 0; echo DONE; }
trap finish EXIT
R0=$(docker inspect -f "{{.RestartCount}}" fnext-vllm); X0=$(sudo dmesg | grep -c "NVRM: Xid")

steps(){ python3 -I - <<'P'
import json, re, time, urllib.request
B = "http://localhost:8036"
body = {"model": "vllm/qwen3.8-flash-next", "temperature": 0, "max_tokens": 2048,
        "messages": [{"role": "user", "content": "Write a complete red-black tree implementation in C with insert, delete and search, explaining each function."}]}
def drafts():
    t = urllib.request.urlopen(B + "/metrics").read().decode()
    return sum(float(x) for x in re.findall(r"^vllm:spec_decode_num_drafts_total\{[^}]*\} ([0-9.e+]+)", t, re.M))
rs = []
for i in range(4):
    d0 = drafts(); a = time.time()
    r = urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(r, timeout=120).read()
    rs.append((drafts() - d0) / (time.time() - a))
print("steps/s " + " ".join("%.1f" % x for x in rs))
P
}

for level in "0 0 0 0" "50 190 145 205" "100 240 195 255" "150 290 245 305" "200 340 295 355" "250 390 345 405"; do
  echo "== offsets $level ($(date +%T))"; $OC $level >/dev/null
  sleep 2
  steps
  nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits | tr '\n' ' '; echo "MHz (idle after run)"
  bad=0
  for run in 1 2 3 4 5; do
    out=$(timeout 200 python3 -I bangcheck.py 2>&1); rc=$?
    line=$(echo "$out" | grep "^generations"); echo "  $line"
    echo "$out" | grep "^BAD" | head -3 | cut -c1-160
    c=$(nvidia-smi --query-gpu=clocks.gr,temperature.gpu,power.draw --format=csv,noheader,nounits | tr '\n' ';')
    [ "$run" = 3 ] && echo "  under load: $c"
    x=$(sudo dmesg | grep -c "NVRM: Xid"); r=$(docker inspect -f "{{.RestartCount}}" fnext-vllm)
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$rc" -ne 0 ] && { echo "STOP rc=$rc"; exit 1; }
    [ $((x-X0)) -gt 0 ] && { echo STOP-xid; sudo dmesg | grep "NVRM: Xid" | tail -5 | cut -c1-180; exit 1; }
    [ $((r-R0)) -gt 0 ] && { echo STOP-restart; exit 1; }
    [ "$t" -ge 85 ] && { echo "STOP-temp $t"; exit 1; }
    echo "$line" | grep -q "corrupt 0" || bad=1
    [ $bad = 1 ] && { echo STOP-corrupt; exit 1; }
  done
  chk=$(timeout 200 python3 bench.py check 2>&1 | tail -1); echo "  $chk"
  [[ "$chk" == *"8/8"* ]] || { echo STOP-check; exit 1; }
  echo "  level passed ($(date +%T))"
done
