#!/bin/bash
cd ~/QWEN-PEDIA/recipes/qwen3.8-flash-next
wait_h(){ for i in $(seq 1 $1); do [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:8036/health)" = 200 ] && { echo "healthy $(date +%T)"; return 0; }; sleep 10; done; echo "NOT healthy $(date +%T)"; return 1; }
docker rm -f fnext-vllm >/dev/null; echo "== fresh container $(date +%T)"; ./launch.sh | tail -1
if wait_h 72; then cd ~/bench; python3 bench.py check; python3 bench.py decode | cut -c1-110; fi
echo "== restart same container (autotune cache path) $(date +%T)"; docker restart -t 30 fnext-vllm >/dev/null
if wait_h 60; then cd ~/bench; python3 bench.py check; python3 bench.py decode | cut -c1-110; else nvidia-smi --query-gpu=index,pstate,utilization.gpu,power.draw --format=csv,noheader; docker logs --tail 3 fnext-vllm 2>&1 | cut -c1-140; fi
echo "xid $(sudo dmesg | grep -ciE 'nvrm: xid')"; echo DONE
