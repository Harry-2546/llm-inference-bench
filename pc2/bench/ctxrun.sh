#!/bin/bash
# Same long-context prompts on r10, then on Basalt with GPUs 0,1; r10 is restored at the end.
cd ~/bench
for p in $(pgrep -x sleep); do [ "$(tr -d '\0' </proc/$p/cmdline)" = "sleep7200" ] && kill "$p"; done
rm -f ~/basalt-watchdog.pid
end=$(( $(date +%s)+1500 ))
while :; do
  last=$(docker logs -t fnext-vllm 2>&1 | grep "POST /v1" | grep -v "127.0.0.1\|172.17" | tail -1 | cut -c1-19)
  age=$(( $(date +%s) - $(date -d "${last}Z" +%s) ))
  run=$(curl -s -m 5 localhost:8036/metrics | grep -E "^vllm:num_requests_running" | awk '{print $2}')
  [ "$age" -ge 900 ] && [ "$run" = "0.0" ] && break
  [ $(date +%s) -ge $end ] && { echo "NOT-IDLE last=$last, nothing stopped"; exit 0; }
  sleep 20
done
echo "===== r10 ($(date +%T)), last outside $last"
python3 -I ctxcmp.py 8036 2>&1 | cut -c1-130
last2=$(docker logs -t fnext-vllm 2>&1 | grep "POST /v1" | grep -v "127.0.0.1\|172.17" | tail -1 | cut -c1-19)
[ "$last2" = "$last" ] || { echo "outside request at $last2 during r10 run, stopping here"; exit 0; }
docker stop -t 60 fnext-vllm >/dev/null
setsid nohup bash -c 'sleep 3600; pkill -f bin/basalt; sleep 5; docker start fnext-vllm' >/dev/null 2>&1 < /dev/null &
WD=$!
sudo prlimit --memlock=unlimited:unlimited --pid $$
finish(){ pkill -f bin/basalt 2>/dev/null; sleep 6; pkill -9 -f bin/basalt 2>/dev/null; k=$(pgrep -P $WD); kill $WD 2>/dev/null; kill $k 2>/dev/null; docker start fnext-vllm >/dev/null; for i in $(seq 1 60); do curl -sf -m 3 localhost:8036/health >/dev/null && { echo "r10 healthy $(date +%T)"; break; }; sleep 10; done; pgrep -af "sleep 3600" | wc -l; echo DONE; }
trap finish EXIT
sleep 3
( cd ~/basalt; exec bin/basalt serve --config ~/bench/bt/cfg1.json --python ~/basalt/.venv/bin/python ) > bt/s3.log 2>&1 &
for i in $(seq 1 60); do
  c=$(curl -s -m 600 -o /dev/null -w "%{http_code}" localhost:8037/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"vllm/qwen3.8-flash-next","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}')
  [ "$c" = 200 ] && break; sleep 5
done
echo "===== Basalt 2 cards ($(date +%T)) http=$c"
python3 -I ctxcmp.py 8037 2>&1 | cut -c1-130
grep -E "done " bt/s3.log | tail -3 | cut -c1-140
grep -E "warning|error" bt/s3.log | tail -4 | cut -c1-170
