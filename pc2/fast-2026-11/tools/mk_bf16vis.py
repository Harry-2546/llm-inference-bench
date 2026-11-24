#!/usr/bin/env python3
"""mk_bf16vis.py SRC_SNAPSHOT VIS_SNAPSHOT OUT : OUT = SRC weights (relative symlinks)
+ VIS's unquantized BF16 vision tower (visual-bf16.safetensors).
SRC: FP4-Local-format checkpoint (quantized vision: FP8 attn + W4A16 MLP, which needs --mm-encoder-tp-mode data on TP4)
VIS: nvidia/Qwen3.8-Flash-Next-NVFP4 snapshot (BF16 vision, same module names).
Afterwards run strip_vis_shard.py OUT: vLLM loads whole shard files, so the shard that still holds SRC's
visual tensors must be rewritten without them.
config.json: drop model.visual.* from quantized_layers, add 'model.visual*' to ignore.
index: visual tensors point to the new file. Original snapshots are not modified."""
import glob, json, os, struct

import sys
dd, nv, out = (os.path.abspath(os.path.expanduser(a)) for a in sys.argv[1:4])
assert not os.path.exists(out), out + " exists"
os.makedirs(out)
VIS = "visual-bf16.safetensors"

# 1) symlink everything except the two files we rewrite
for e in os.listdir(dd):
    if e in ("config.json", "model.safetensors.index.json"):
        continue
    os.symlink(os.path.relpath(os.path.join(dd, e), out), os.path.join(out, e))

# 2) collect NVIDIA visual tensors (raw bytes)
tensors = {}
for f in glob.glob(nv + "/**/*.safetensors", recursive=True):
    with open(f, "rb") as fh:
        n8 = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n8))
        for n, v in h.items():
            if n != "__metadata__" and n.startswith("model.visual."):
                a, b = v["data_offsets"]
                fh.seek(8 + n8 + a)
                tensors[n] = (v["dtype"], v["shape"], fh.read(b - a))
assert len(tensors) == 333, len(tensors)
hdr, off = {"__metadata__": {"format": "pt"}}, 0
for n in sorted(tensors):
    dt, sh, data = tensors[n]
    hdr[n] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + len(data)]}
    off += len(data)
hb = json.dumps(hdr, separators=(",", ":")).encode()
hb += b" " * ((8 - len(hb) % 8) % 8)
with open(os.path.join(out, VIS), "wb") as fo:
    fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
    for n in sorted(tensors):
        fo.write(tensors[n][2])

# 3) index: drop SRC visual entries, add new file
idx = json.load(open(os.path.join(dd, "model.safetensors.index.json")))
wm = idx["weight_map"]
dropped = [k for k in wm if k.startswith("model.visual.")]
for k in dropped:
    del wm[k]
for n in tensors:
    wm[n] = VIS
json.dump(idx, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)

# 4) config: visual unquantized
cfg = json.load(open(os.path.join(dd, "config.json")))
q = cfg["quantization_config"]
ql = q["quantized_layers"]
vis_q = [k for k in ql if k.startswith("model.visual.")]
for k in vis_q:
    del ql[k]
q["ignore"] = q.get("ignore", []) + ["model.visual*"]
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

print(f"visual tensors {len(tensors)} ({off/2**30:.3f} GiB); dropped index entries {len(dropped)}; "
      f"dropped quantized_layers {len(vis_q)}; remaining quantized_layers {len(ql)}")
