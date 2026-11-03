#!/bin/bash
cd ~/bench
until curl -sf localhost:8036/health >/dev/null; do sleep 10; done
echo "== healthy $(date +%T)"
sudo docker logs fnext-vllm 2>&1 | grep -E "GPU KV cache size" | tail -1
echo "-- needle ~250K"; python3 needle.py 250000
echo "-- needle ~480K"; python3 needle.py 480000
echo "-- check"; python3 bench.py check
echo "-- step short"; python3 bench_strata_r10.py step
echo "xid=$(sudo dmesg | grep -ciE xid)"; echo DONE
