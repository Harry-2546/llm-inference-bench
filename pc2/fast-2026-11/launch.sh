#!/bin/bash
# Qwen3.8-Flash-Next FP4-Local on 4x RTX 5090 (TP4), upstream vLLM nightly 8586ccb + mounted patches.
# Serving config as of 2026-11-21 (see README.md in this directory for the per-change measurements).
#
# Prereqs:
#   patches/build_patches.sh "$PATCH_DIR"     # 13 patched vLLM files (#66277, #68472-#68474, local KV-offload fix)
#   MODEL: either the hub id (quantized vision -> VISION=quant), or a dir built with
#          tools/mk_bf16vis.py + tools/strip_vis_shard.py (BF16 vision -> VISION=bf16, the default here).
#   ~125 GB host RAM: n-gram embedding (~27 GiB) lives in pinned host memory, plus 56 GB CPU KV tier.
set -euo pipefail
IMG=vllm/vllm-openai:nightly-8586ccbcf58e69ecb69e370dd2fa69fb739f9b0c
SITE=/usr/local/lib/python3.14/dist-packages
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-$HERE/patches/out}"
MODEL="${MODEL:-/root/.cache/huggingface/fp4local-bf16vis}"
VISION="${VISION:-bf16}"
NAME="${NAME:-fnext}"
PORT="${PORT:-8036}"
CTX=524288   # YaRN 2x on model and drafter

MOUNTS=()
while IFS= read -r -d '' f; do
  rel="${f#"$PATCH_DIR"/}"; MOUNTS+=(-v "$f:$SITE/$rel:ro")
done < <(find "$PATCH_DIR/vllm" -name '*.py' -print0)
[ "${#MOUNTS[@]}" -eq 26 ] || { echo "expected 13 patched files in $PATCH_DIR, got $((${#MOUNTS[@]}/2))"; exit 1; }

VIS_ARGS=()
# Quantized vision (FP8 attn + W4A16 MLP) breaks the default TP 'weights' mode on TP4
# ("features size is not multiple of 16"); run the encoder data-parallel instead.
[ "$VISION" = quant ] && VIS_ARGS=(--mm-encoder-tp-mode data
  --kernel-config '{"linear_backend_per_quant": {"nvfp4_w4a16": "auto"}}')

SPEC="{\"method\":\"qwen4_exp_mtp\",\"num_speculative_tokens\":3,\"max_model_len\":$CTX,\"moe_backend\":\"b12x\",\"draft_sample_method\":\"probabilistic\",\"draft_lm_head_quantization\":\"nvfp4\"}"
YARN='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'

# expandable_segments:True needs the local KV-offload patch (upstream rejects it with any KV connector);
# it removes the fragmentation that caused a 512 MiB QSA-indexer workspace OOM with ~1-2 GB headroom/card.
docker run -d --name "$NAME" --restart unless-stopped --gpus all --ipc host --shm-size 64m \
  --ulimit memlock=-1:-1 -p "$PORT":8000 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_SIMPLE_KV_OFFLOAD=1 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 -e VLLM_CUSTOM_AR_ALLOW_PCIE=1 -e VLLM_CUSTOM_ALL_GATHER=1 \
  -e NCCL_PROTO=Simple \
  -e VLLM_DISABLE_EAGLE_BLOCK_DROP=1 -e VLLM_MOE_USE_DEEP_GEMM=0 -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_GDN_FUSE_IN_PROJ=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  "${MOUNTS[@]}" \
  --entrypoint vllm "$IMG" serve \
  --model "$MODEL" --served-model-name vllm/qwen3.8-flash-next \
  --host 0.0.0.0 --port 8000 \
  --quantization modelopt_mixed \
  --tensor-parallel-size 4 --async-scheduling --max-num-seqs 8 \
  --max-model-len "$CTX" --hf-overrides "$YARN" \
  --kv-cache-dtype fp8 --kv-cache-memory 4800000000 \
  --kv-offloading-size 56 --kv-offloading-backend native \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --engram-config '{"cpu_offload": true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,24,32]}' \
  --speculative-config "$SPEC" \
  --moe-backend b12x --linear-backend b12x \
  --max-num-batched-tokens 4096 \
  --quantization-config '{"linear":{"phase":{"decode":"mxfp8","prefill":"mxfp8"}}}' \
  "${VIS_ARGS[@]}"

echo "launched $NAME on :$PORT"
