#!/bin/bash
# Memory-offset sweep under 8-way concurrent long-context load. Stops on check failure or Xid.
cd ~/bench
for off in 0 4000 5000 6000; do
  echo "== NVML +$off ($(date +%T))"; sudo ~/hfenv/bin/python memoc.py $off | head -1
  python3 bench.py conc conc
  chk=$(python3 bench.py check); echo "$chk"
  x=$(sudo dmesg | grep -ciE xid); echo "xid=$x"
  [ "$x" -gt 0 ] && { echo STOP-xid; break; }
  [[ "$chk" == *"8/8"* ]] || { echo STOP-check; break; }
done
echo "== final: setting +4000 (last verified)"; sudo ~/hfenv/bin/python memoc.py 4000 | head -1
echo DONE
