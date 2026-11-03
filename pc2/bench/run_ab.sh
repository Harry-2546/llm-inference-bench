#!/bin/bash
# usage: run_ab.sh <label>
cd ~/bench; L=$1
until curl -sf localhost:8036/health >/dev/null; do sleep 10; done
echo "== $L healthy $(date +%T)"
python3 bench.py check
python3 ab.py
echo "xid=$(sudo dmesg | grep -ciE 'nvrm: xid')"; echo DONE
