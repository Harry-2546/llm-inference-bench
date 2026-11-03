#!/bin/bash
# Core-offset sweep on r10. Stops on Xid, check failure, timeout or container restart; always ends at offset 0.
cd ~/bench
OC="sudo /opt/gpu-oc/bin/python /tmp/gpuoc.py"
finish(){ echo "== reset core offset 0 ($(date +%T))"; $OC 0; echo DONE; }
trap finish EXIT
R0=$(docker inspect -f "{{.RestartCount}}" fnext-vllm); X0=$(sudo dmesg | grep -ciE "nvrm: xid")
for off in 0 100 150 200 250 300; do
  echo "== core +$off ($(date +%T))"; $OC $off | head -1
  L=/tmp/core-$off.csv
  timeout 400 nvidia-smi --query-gpu=index,utilization.gpu,clocks.gr,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits -l 1 > $L &
  LP=$!
  out=$(timeout 400 python3 bench.py decode conc check 2>&1); rc=$?
  kill $LP 2>/dev/null; wait $LP 2>/dev/null
  echo "$out" | cut -c1-170
  awk -F', *' '$2>=95{n[$1]++;c[$1]+=$3;p[$1]+=$5;if($5>mp[$1])mp[$1]=$5;if($6>mt[$1])mt[$1]=$6;if($7!="0x0000000000000000")th[$1]=$7}END{for(i=0;i<4;i++)printf "  GPU%d busy-samples %d avg %.0f MHz, avg %.0f W, peak %.0f W, peak %d C, throttle %s\n",i,n[i],c[i]/n[i],p[i]/n[i],mp[i],mt[i],(th[i]?th[i]:"none")}' $L
  x=$(sudo dmesg | grep -ciE "nvrm: xid"); r=$(docker inspect -f "{{.RestartCount}}" fnext-vllm)
  echo "rc=$rc xid_new=$((x-X0)) restarts_new=$((r-R0))"
  [ $rc -eq 0 ] || { echo STOP-rc; break; }
  [ $((x-X0)) -gt 0 ] && { echo STOP-xid; sudo dmesg | grep -iE "nvrm: xid" | tail -5 | cut -c1-200; break; }
  [ $((r-R0)) -gt 0 ] && { echo STOP-restart; break; }
  [[ "$out" == *"8/8"* ]] || { echo STOP-check; break; }
done
