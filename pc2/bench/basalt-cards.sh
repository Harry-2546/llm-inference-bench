#!/bin/bash
# Stop r10, run Basalt's community preset on 1, 2, 3 and 4 cards. r10 stays stopped for the next test;
# the watchdog brings it back after 2 h if nothing cancels it.
cd ~/basalt
last=$(docker logs -t fnext-vllm 2>&1 | grep "POST /v1" | grep -v "127.0.0.1\|172.17" | tail -1 | cut -c1-19)
echo "last outside request $last, now $(date +%T)"
docker stop -t 60 fnext-vllm >/dev/null && echo "r10 stopped"
setsid nohup bash -c 'sleep 7200; pkill -f bin/basalt; sleep 5; docker start fnext-vllm' >/dev/null 2>&1 < /dev/null &
echo $! > ~/basalt-watchdog.pid
sleep 3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';'; echo
sudo prlimit --memlock=unlimited:unlimited --pid $$
echo "memlock $(ulimit -l)"
X0=$(sudo dmesg | grep -c "NVRM: Xid")
F=/tmp/basalt-community.json
for cards in 0 0,1 0,1,2 0,1,2,3; do
  echo "===== cards $cards ($(date +%T))"
  CUDA_VISIBLE_DEVICES=$cards timeout 1200 bin/basalt-bench --community --model models/Qwen3.8-FN-NVFP4.basalt --submitter local --community-file $F > /tmp/bb-$cards.log 2>&1
  echo "rc=$? xid_new=$(( $(sudo dmesg | grep -c "NVRM: Xid")-X0 ))"
  grep -vE "^\s*$" /tmp/bb-$cards.log | tail -26 | cut -c1-170
  nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader | tr '\n' ';'; echo
done
echo DONE
