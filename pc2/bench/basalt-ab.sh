#!/bin/bash
# Same prompts as the r10 runs against Basalt on GPUs 0,1 at :8037 (localhost). Ends by restarting r10 and cancelling the watchdog.
cd ~/bench; rm -rf bt; mkdir bt
end=$(( $(date +%s)+3000 ))
while :; do
  last=$(docker logs -t fnext-vllm 2>&1 | grep "POST /v1" | grep -v "127.0.0.1\|172.17" | tail -1 | cut -c1-19)
  age=$(( $(date +%s) - $(date -d "${last}Z" +%s) ))
  run=$(curl -s -m 5 localhost:8036/metrics | grep -E "^vllm:num_requests_running" | awk '{print $2}')
  [ "$age" -ge 900 ] && [ "$run" = "0.0" ] && break
  [ $(date +%s) -ge $end ] && { echo "NOT-IDLE last=$last, r10 left running"; exit 0; }
  sleep 20
done
echo "idle since $last, stopping r10 $(date +%T)"
docker stop -t 60 fnext-vllm >/dev/null
setsid nohup bash -c 'sleep 7200; pkill -f bin/basalt; sleep 5; docker start fnext-vllm' >/dev/null 2>&1 < /dev/null &
echo $! > ~/basalt-watchdog.pid
sleep 3
for f in accept ab bench bangcheck; do sed 's/8036/8037/g' $f.py > bt/$f.py; done
python3 -I - <<'P'
import re
for f in ("accept", "ab", "bench", "bangcheck"):
    p = f"bt/{f}.py"; s = open(p).read()
    s = s.replace('urllib.request.urlopen(B + "/metrics")', 'urllib.request.urlopen(urllib.request.Request(B + "/metrics", headers={"Accept": "text/plain"}))')
    s = s.replace('urllib.request.urlopen(B+"/metrics")', 'urllib.request.urlopen(urllib.request.Request(B+"/metrics", headers={"Accept": "text/plain"}))')
    s = s.replace('urllib.request.urlopen(METRICS, timeout=10)', 'urllib.request.urlopen(urllib.request.Request(METRICS, headers={"Accept": "text/plain"}), timeout=10)')
    s = s.replace('dl.get("reasoning")', '(dl.get("reasoning") or dl.get("reasoning_content"))')
    s = s.replace('msg.get("reasoning")', '(msg.get("reasoning") or msg.get("reasoning_content"))')
    open(p, "w").write(s)
P
X0=$(sudo dmesg | grep -c "NVRM: Xid")
sudo prlimit --memlock=unlimited:unlimited --pid $$
SRV=""
stop_srv(){ [ -n "$SRV" ] && kill "$SRV" 2>/dev/null; pkill -f bin/basalt 2>/dev/null; sleep 6; pkill -9 -f bin/basalt 2>/dev/null; SRV=""; sleep 2; }
finish(){ stop_srv; echo "== restore r10 ($(date +%T))"; docker start fnext-vllm >/dev/null; [ -f ~/basalt-watchdog.pid ] && pkill -P "$(cat ~/basalt-watchdog.pid)" 2>/dev/null; kill "$(cat ~/basalt-watchdog.pid)" 2>/dev/null; rm -f ~/basalt-watchdog.pid; for i in $(seq 1 60); do curl -sf -m 3 localhost:8036/health >/dev/null && { echo "r10 healthy $(date +%T)"; break; }; sleep 10; done; echo "xid_new=$(( $(sudo dmesg | grep -c 'NVRM: Xid')-X0 ))"; echo DONE; }
trap finish EXIT
start_srv(){ # parallel log
  cat > ~/bench/bt/cfg.json <<E
{"server": {"model": "$HOME/basalt/models/Qwen3.8-FN-NVFP4.basalt", "gpu": "0,1", "port": 8037, "alias": "vllm/qwen3.8-flash-next", "parallel": $1},
 "engine": {"expert-cache": "auto", "prefill": "auto", "kv": "int8", "resident-experts": true, "max-context": 524288}}
E
  ( cd ~/basalt; exec bin/basalt serve --config ~/bench/bt/cfg.json --python ~/basalt/.venv/bin/python ) > "$2" 2>&1 &
  SRV=$!
  for i in $(seq 1 90); do
    c=$(curl -s -m 600 -o /dev/null -w "%{http_code}" localhost:8037/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"vllm/qwen3.8-flash-next","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}')
    [ "$c" = 200 ] && { echo "ready $(date +%T)"; grep -E "^\[server\] (engine|auto|loaded|kv|slots|weights)" "$2" | cut -c1-170; return 0; }
    sleep 5
  done
  echo "server not ready"; tail -20 "$2" | cut -c1-170; exit 1
}
cd bt
echo "===== single stream ($(date +%T))"; start_srv 1 s1.log
python3 bench.py check 2>&1 | tail -1
echo "-- accept"; python3 -I accept.py 2>&1 | tail -18
echo "-- ab"; python3 ab.py 2>&1 | grep -vE "^(needle|order) [0-9]" | cut -c1-150
grep -E "^order|^needle" /dev/null
echo "-- tool call"
python3 -I - <<'P'
import json, urllib.request
body = {"model": "vllm/qwen3.8-flash-next", "temperature": 1.0, "max_tokens": 2000, "messages": [{"role": "user", "content": "What is the weather in Osaka right now? Use the tool."}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
r = urllib.request.Request("http://localhost:8037/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=120))
c = d["choices"][0]; m = c["message"]
print("finish", c["finish_reason"], "keys", sorted(m), "tool_calls", json.dumps(m.get("tool_calls"))[:200])
msgs = body["messages"] + [m, {"role": "tool", "tool_call_id": m["tool_calls"][0]["id"], "content": "{\"temp_c\": 21, \"sky\": \"clear\"}"}]
body2 = dict(body, messages=msgs)
r = urllib.request.Request("http://localhost:8037/v1/chat/completions", json.dumps(body2).encode(), {"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=120))
print("second", d["choices"][0]["finish_reason"], repr((d["choices"][0]["message"].get("content") or "")[:120]))
P
grep -E "done " s1.log | tail -6 | cut -c1-150
nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv,noheader | tr '\n' ';'; echo
stop_srv
echo "===== parallel 8 ($(date +%T))"; start_srv 8 s8.log
python3 bench.py check 2>&1 | tail -1
for i in 1 2 3; do python3 -I bangcheck.py 2>&1 | grep -E "^generations|^BAD|^short|busy" | cut -c1-150; done
grep -E "warning|error" s8.log | tail -5 | cut -c1-170
