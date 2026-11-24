#!/bin/bash
# build_patches.sh OUT : rebuild the patched vLLM files mounted read-only over the 8586ccb nightly image.
#   upstream/pr68474.diff  vllm#68472-#68474 stack (phase-split MXFP8 activations), as of 2026-11-21
#   upstream/pr66277.diff  vllm#66277 (b12x kernels unloaded under concurrency -> illegal memory access)
#   fix68474.py / fix66277imports.py  hand-apply the hunks that do not apply cleanly on 8586ccb
#   local/kv-offload-expandable-segments.diff  allow expandable_segments:True with SimpleCPUOffloadConnector (cf. vllm#42405)
# OUT receives only the changed files, laid out as vllm/...; mount each over dist-packages/vllm/...
set -euo pipefail
IMG=vllm/vllm-openai:nightly-8586ccbcf58e69ecb69e370dd2fa69fb739f9b0c
SITE=/usr/local/lib/python3.14/dist-packages
OUT=$(realpath -m "${1:?usage: build_patches.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cid=$(docker create "$IMG"); docker cp -q "$cid:$SITE/vllm" "$W/orig"; docker rm "$cid" >/dev/null
cp -a "$W/orig" "$W/vllm"
cd "$W"
git init -q .
# --reject: apply what applies, leave .rej for the hunks the fix scripts handle
git apply --include='vllm/*' --reject "$HERE/upstream/pr68474.diff" 2>&1 | grep -E '^(Rejected|error)' || true
python3 "$HERE/fix68474.py" "$W"
git apply --include='vllm/*' --reject "$HERE/upstream/pr66277.diff" 2>&1 | grep -E '^(Rejected|error)' || true
python3 "$HERE/fix66277imports.py" "$W/vllm/utils/b12x.py"
git apply "$HERE/local/kv-offload-expandable-segments.diff"
find vllm -name '*.rej' -delete; find vllm -name '*.orig' -delete
mkdir -p "$OUT"
(cd vllm && find . -name '*.py' -print0) | while IFS= read -r -d '' f; do
  if ! cmp -s "vllm/$f" "orig/$f"; then mkdir -p "$OUT/vllm/$(dirname "$f")"; cp "vllm/$f" "$OUT/vllm/$f"; echo "patched vllm/${f#./}"; fi
done
