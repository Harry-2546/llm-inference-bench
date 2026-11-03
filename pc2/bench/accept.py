"""Acceptance and speed by prompt type, temperature and thinking. Temperature 0 and 1 use the same prompts."""
import json, re, time, urllib.request
B = "http://localhost:8036"
PROMPTS = {
    "code": "Write a complete red-black tree implementation in C with insert, delete and search, explaining each function.",
    "prose": "Write a long essay about how the printing press changed European society.",
    "zh": "请用中文详细讲解 TCP 拥塞控制的发展历史和主要算法。",
    "copy": "Repeat the following text exactly, then repeat it once more:\n" + "\n".join(f"def handler_{i}(request): return respond(request, status={200 + i})" for i in range(40)),
}


def metrics():
    t = urllib.request.urlopen(B + "/metrics").read().decode()
    o = {}
    for k in ("num_drafts", "num_accepted_tokens", "num_draft_tokens"):
        o[k] = sum(float(x) for x in re.findall(r"^vllm:spec_decode_" + k + r"_total\{[^}]*\} ([0-9.e+]+)", t, re.M))
    return o


def one(prompt, temp, think):
    body = {"model": "vllm/qwen3.8-flash-next", "temperature": temp, "top_p": 0.95 if temp else 1.0, "max_tokens": 768,
            "messages": [{"role": "user", "content": prompt}], "chat_template_kwargs": {"enable_thinking": think}}
    m0 = metrics(); a = time.time()
    r = urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    u = json.load(urllib.request.urlopen(r, timeout=120))["usage"]
    w = time.time() - a; m1 = metrics()
    st = m1["num_drafts"] - m0["num_drafts"]
    acc = (m1["num_accepted_tokens"] - m0["num_accepted_tokens"]) / max(1, m1["num_draft_tokens"] - m0["num_draft_tokens"])
    return u["completion_tokens"] / w, acc, u["completion_tokens"] / max(1, st)


print(f"{'prompt':6} {'temp':>4} {'think':>5} | {'tok/s':>6} {'accept':>6} {'tok/step':>8}")
for name, p in PROMPTS.items():
    for temp, think in ((0, True), (1.0, True), (0, False), (1.0, False)):
        rs = [one(p, temp, think) for _ in range(3)]
        print(f"{name:6} {temp:4} {str(think):>5} | {sum(r[0] for r in rs) / 3:6.0f} {sum(r[1] for r in rs) / 3:6.3f} {sum(r[2] for r in rs) / 3:8.2f}")
