#!/bin/bash
cd ~/bench
echo "== r10 on PC2 (4x5090 TP4+EP, mem +4000), $(date +%F\ %T)"
echo "-- step short"; python3 bench_strata_r10.py step
echo "-- step ~32K"; python3 bench_strata_r10.py step 32000
echo "-- step ~108K"; python3 bench_strata_r10.py step 107500
echo "-- prefill"; python3 bench_strata_r10.py prefill
echo "xid=$(sudo dmesg | grep -ciE xid)"
echo DONE
