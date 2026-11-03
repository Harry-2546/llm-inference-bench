#!/bin/bash
cd ~/bench
echo "== matched ctx (23000 / 77000 words = Strata 32K / 108K), $(date +%T)"
echo "-- step ~32K"; python3 bench_strata_r10.py step 23000
echo "-- step ~108K"; python3 bench_strata_r10.py step 77000
echo "== concurrency"; python3 conc_strata.py
echo "xid=$(sudo dmesg | grep -ciE xid)"; echo DONE
