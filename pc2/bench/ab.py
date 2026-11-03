"""Fixed-seed A/B set: decode speed at short and >native context, needle recall, order recall."""
import json, random, re, sys, time, urllib.request
B="http://localhost:8036"; M="vllm/qwen3.8-flash-next"
W=("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
   "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
def metrics():
    t=urllib.request.urlopen(B+"/metrics").read().decode(); o={}
    for k in ("num_drafts","num_accepted_tokens","num_draft_tokens"):
        o[k]=sum(float(x) for x in re.findall(r"^vllm:spec_decode_"+k+r"_total\{[^}]*\} ([0-9.e+]+)",t,re.M))
    return o
def call(text,max_tokens):
    body={"model":M,"messages":[{"role":"user","content":text}],"max_tokens":max_tokens,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}
    r=urllib.request.Request(B+"/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"})
    a=time.time(); first=None; u=None; out=[]
    for line in urllib.request.urlopen(r,timeout=3600):
        line=line.decode().strip()
        if not line.startswith("data: ") or line=="data: [DONE]": continue
        d=json.loads(line[6:])
        if d.get("usage"): u=d["usage"]
        if d.get("choices"):
            dl=d["choices"][0]["delta"]
            if dl.get("content"): out.append(dl["content"])
            if (dl.get("content") or dl.get("reasoning")) and first is None: first=time.time()
    return "".join(out),u,first-a,time.time()-first
def decode(label,text,n=3,mt=768):
    rs=[]
    for i in range(n):
        m0=metrics(); s,u,tt,dt=call(text,mt); m1=metrics()
        st=m1["num_drafts"]-m0["num_drafts"]; ac=(m1["num_accepted_tokens"]-m0["num_accepted_tokens"])/max(1,m1["num_draft_tokens"]-m0["num_draft_tokens"])
        rs.append((u["completion_tokens"]/dt,st/dt,ac,tt,u["prompt_tokens"]))
    avg=lambda k:sum(r[k] for r in rs)/len(rs)
    print(f"[{label:14s}] prompt {rs[0][4]:7d} | tok/s {avg(0):6.1f} | steps/s {avg(1):6.1f} | accept {avg(2):.3f} | TTFT first {rs[0][3]:6.2f}s last {rs[-1][3]:6.2f}s",flush=True)
def filler(n,seed):
    r=random.Random(seed); return [r.choice(W) for _ in range(n)]
CODE="Write a complete Python implementation of a red-black tree with insert, delete, search, in-order iteration, and a pytest test suite. Output only code."
PROSE="Write a long, detailed essay on the history of the Silk Road, its economic effects, and its cultural legacy."
ZH="请详细写一篇长文，介绍中国古代四大发明的历史、传播过程以及对世界文明的影响。"
print("== short-context decode",flush=True)
decode("code",CODE); decode("prose",PROSE); decode("zh",ZH)
N=int(440000/1.4)   # ~483K tokens
f=filler(N,20261031); ft=" ".join(f)
print("== decode past native",flush=True)
decode("code@483K",ft+"\n\nIgnore the noise above. "+CODE,n=3)
decode("prose@483K",ft+"\n\nIgnore the noise above. "+PROSE,n=3)
print("== needle recall",flush=True)
ok=0;tot=0
for tgt,depths in ((int(250000/1.4),(0.1,0.5,0.9)),(N,(0.1,0.3,0.5,0.7,0.9))):
    for d in depths:
        r=random.Random(int(tgt*d)); code=f"{r.randint(1000,9999)}-{r.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{r.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"; city=r.choice(["Lisbon","Osaka","Quito","Tallinn","Hobart","Windhoek","Reykjavik","Cusco"])
        w=filler(tgt,tgt); w.insert(int(tgt*d),f" The secret vault code is {code} and the vault is in {city}. ")
        q="\n\nSomewhere in the text above there is one sentence about a secret vault. What is the vault code, and in which city is the vault? Answer in one short line."
        s,u,tt,dt=call(" ".join(w)+q,4000); p=code in s and city in s; ok+=p; tot+=1
        print(f"needle {u['prompt_tokens']:7d} tok depth {d:.0%} {'PASS' if p else 'FAIL'} comp {u['completion_tokens']} | {s.strip()[:90]!r}",flush=True)
print(f"needle total {ok}/{tot}",flush=True)
print("== order recall (position-sensitive)",flush=True)
ok=0;tot=0
for (a,b) in ((0.05,0.15),(0.40,0.50),(0.55,0.65),(0.85,0.95),(0.15,0.05),(0.50,0.40),(0.65,0.55),(0.95,0.85)):
    w=list(f); lo,hi=sorted((a,b)); first,second=("RED","BLUE") if a<b else ("BLUE","RED")
    w.insert(int(N*hi),f" Then the {second} door was opened. "); w.insert(int(N*lo),f" Then the {first} door was opened. ")
    q="\n\nTwo sentences in the text above mention a door being opened: a RED door and a BLUE door. Which door was opened first in the text? Answer with one word: RED or BLUE."
    s,u,tt,dt=call(" ".join(w)+q,6000); ans=re.findall(r"RED|BLUE",s.upper()); p=bool(ans) and ans[-1]==first; ok+=p; tot+=1
    print(f"order {a:.2f}/{b:.2f} expect {first} {'PASS' if p else 'FAIL'} comp {u['completion_tokens']} | {s.strip()[:50]!r}",flush=True)
print(f"order total {ok}/{tot}",flush=True)
