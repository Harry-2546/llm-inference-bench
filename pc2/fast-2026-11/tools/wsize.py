"""wsize.py SNAPSHOT_DIR : checkpoint bytes by component (reads safetensors headers only)."""
import json, struct, glob, os, re, sys, collections
d = sys.argv[1]
cat = collections.Counter(); dt = collections.Counter()
def k(n):
    if "engram" in n: return "engram(CPU)"
    if re.search(r"\bmtp\b|^mtp\.|\.mtp\.", n): return "mtp_draft"
    if "visual" in n or "vision" in n: return "vision"
    if "embed_tokens" in n: return "embed"
    if "lm_head" in n: return "lm_head"
    if ".experts." in n: return "moe_experts"
    if "shared_expert" in n: return "moe_shared"
    if "mlp.gate." in n or n.endswith("mlp.gate.weight"): return "moe_router"
    if "linear_attn" in n: return "gdn_linear_attn"
    if "self_attn" in n or "indexer" in n: return "attn(qsa)"
    if "norm" in n: return "norm"
    return "other:" + n.split(".")[2] if n.count(".") > 2 else "other"
for f in glob.glob(d + "/**/*.safetensors", recursive=True):
    with open(f, "rb") as fh:
        h = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
    for n, v in h.items():
        if n == "__metadata__": continue
        s = v["data_offsets"][1] - v["data_offsets"][0]
        cat[k(n)] += s; dt[v["dtype"]] += s
G = 2**30
for c, s in cat.most_common(): print(f"{c:22s} {s/G:7.2f} GiB")
print("total", round(sum(cat.values())/G, 2)); print({a: round(b/G, 2) for a, b in dt.items()})
