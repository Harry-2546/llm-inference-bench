#!/usr/bin/env python3
"""strip_vis_shard.py OUT : in the mk_bf16vis.py output dir replace the 'tensors' dir symlink with a real dir of
per-file relative symlinks, except shards holding model.visual.* tensors, which are rewritten without them."""
import json, os, struct, sys

out = os.path.abspath(os.path.expanduser(sys.argv[1]))
link = os.path.join(out, "tensors")
assert os.path.islink(link)
src = os.path.realpath(link)
os.unlink(link); os.makedirs(link)
for f in sorted(os.listdir(src)):
    sp = os.path.join(src, f)
    with open(sp, "rb") as fh:
        n8 = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n8))
        if not any(k.startswith("model.visual.") for k in h):
            os.symlink(os.path.relpath(sp, link), os.path.join(link, f)); continue
        keep = sorted((k for k in h if k != "__metadata__" and not k.startswith("model.visual.")),
                      key=lambda k: h[k]["data_offsets"][0])
        hdr, off = {"__metadata__": h.get("__metadata__", {"format": "pt"})}, 0
        for k in keep:
            a, b = h[k]["data_offsets"]
            hdr[k] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"], "data_offsets": [off, off + b - a]}
            off += b - a
        hb = json.dumps(hdr, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
        with open(os.path.join(link, f), "wb") as fo:
            fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
            for k in keep:
                a, b = h[k]["data_offsets"]; fh.seek(8 + n8 + a); fo.write(fh.read(b - a))
        print(f"rewrote {f}: kept {len(keep)}, dropped {len(h) - 1 - len(keep)} visual, {off/2**30:.2f} GiB")
