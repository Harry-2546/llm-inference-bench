# FP4-Local speed-up on 4×RTX 5090 (2026-11-21)

Goal: serve the FP4-Local checkpoint of Qwen3.8-Flash-Next faster than our NVFP4 baseline on the same rig, with quality roughly unchanged. One day, about 30 configurations, 1% decision threshold.

Stack: upstream vLLM nightly `8586ccb` (0.34.0rc1.dev1014, FlashInfer 0.7.4) with TP4, `modelopt_mixed` quantization, MTP speculative decoding (3 draft tokens), and YaRN 524K. Three unmerged upstream PRs and one local fix are mounted read-only over the image, which itself stays unmodified.

## Result

| Metric | Final (u24) | NVFP4 baseline | Change |
|---|---|---|---|
| Single stream, think off (warm 3, mean of 3) | 379.3 tok/s | 340 | **+11.6%** |
| 8-way (16 × 1,024 tokens) | 1,375 tok/s | 1,327 | **+3.6%** |
| Cold 60K-token prompt, steady-state TTFT | 4.65 s | ~4.8 s | −3% |
| GSM8K 250 / GPQA 198 | 96.8% / 92.42% | — / 89.9% | |
| wikitext perplexity | 5.204 | 5.277 | |
| 30-min 8-way long-context soak | 473/473 requests, 0 Xid | | |

## What each change was worth

All relative to u6 (upstream + #66277 + MoE W4A16), which averaged 373 single and 1,343 8-way over two runs; its TTFT was ~6.8 s.

| Change | Single | 8-way | TTFT | Kept |
|---|---|---|---|---|
| Draft LM head NVFP4 (`draft_lm_head_quantization`) | +1.2% | +2.4% | | ✅ |
| Vision encoder data-parallel (needed on TP4, see below) | −1.1% | −0.7% | | ✅ |
| `--max-num-batched-tokens 4096` (from 2048) | −3.8% (unexplained) | +1.4% | **−32%** | ✅ |
| Phase-split MXFP8 activations (#68472–#68474) | flat | +1.9% | | ✅ |
| Batch-sharded sampling | −3% | +0.9% when stacked | | ❌ |
| 4 draft tokens (with or without a dynamic count) | −3% | −2 to −7% | | ❌ |
| torch.compile on (model defaults to breakable CUDA graphs) | −2.7% | −4.5% | +0.4 s | ❌ |
| + all-reduce/RMSNorm fusion (requires compile) | −3.1% | −2.5% | | ❌ |
| b12x PCIe all-reduce vs FlashInfer PCIe IPC | −4% | | | ❌ |
| Prefill chunk 6144 / 3072 | | −5% / — | +0 / +3 s | ❌ |
| Draft LM head FP8 | no SM120 kernel, won't start | | | ❌ |

Findings:
1. Upstream `8586ccb` with b12x linear layers hits an illegal memory access at concurrency ≥ 2 (the b12x kernels get unloaded). #66277 fixes it, and without b12x linear layers the model is 4–5% slower.
2. On the upstream image, MTP acceptance matches the NVFP4 checkpoint (2.7–2.8). The low acceptance we saw on a community image was an image problem, not a checkpoint problem.
3. The model defaults to breakable CUDA graphs, so torch.compile and every compile-time fusion are silently off. Turning compile on made it slower.
4. The community image's faster TTFT came from its 6019-token prefill chunk, not from its all-reduce.

## Serving hardening (after the speed run)

| Change | Why | Measured |
|---|---|---|
| CPU KV tier 56 GB (`--kv-offloading-size 56`, 14 GB/rank) | Long agent sessions get evicted from a 594K-token GPU pool | 52,800-token prefix served from CPU: TTFT 4.3 s → 0.45–1.9 s (`tools/offload_check.py`) |
| `--kv-cache-memory` 5.0e9 → 4.8e9 | `--kv-cache-memory` skips memory profiling, so it reserves no headroom | +0.3 GiB/card headroom, still 1.13× one 524K request |
| BF16 vision tower instead of the quantized one | Quantized vision (FP8 attention + W4A16 MLP) breaks the default TP "weights" encoder mode (`features size is not multiple of 16`) | Original-precision vision, −0.09 GiB/card, speed unchanged |
| `expandable_segments:True` + local patch | A 512 MiB QSA-indexer workspace OOMed on a card with 400 MB free plus 543 MB fragmented | 0 OOM since; single 361.4 / 8-way 1,375 / 4.65 s (3 runs) |

**Why the local patch exists.** vLLM refuses `expandable_segments:True` with any KV connector unless `--enable-cumem-allocator` is also set, and in our setup the cumem allocator OOMs during weight loading. For `SimpleCPUOffloadConnector` the check is a false positive: the connector records the KV tensors' `data_ptr` and copies with `cuMemcpyBatchAsync`, and the host side is plain `cudaHostRegister`, so there is no RDMA registration for a VMM remap to break. Upstream discussion: vllm#42405. The patch is a five-line exemption in `_verify_kv_transfer_compat`; see `patches/local/`.

The single-stream mean after the patch, 361.4, is 3% below one unpatched run of 373.8. That is inside run-to-run noise and was not A/B-alternated.

**Per-GPU memory (31.40 GiB visible):**

| Item | GiB |
|---|---|
| CUDA context, NCCL, pre-load | ~1.5 |
| Weights | 22.78 |
| KV cache (fp8, 594,094 tokens) | 4.47 |
| CUDA graphs | 0.24 |
| Activations and free | the rest (~31.5 GB used under load) |

The 95 GiB checkpoint by component (`tools/wsize.py`): MoE experts 59.9, n-gram embedding 26.9 (kept in pinned host RAM by `engram cpu_offload`), GDN 2.0, MTP 1.5, hyper-connections 1.2, embed 1.2, lm_head 1.2, QSA attention 0.6, vision 0.3.

## Measurement notes

- **Noise is bigger than the threshold.** The same config measured twice gave 8-way results 4% apart, and three single-stream runs spread 7%. Decisions at 1% need alternated repeats.
- **Warm-up matters.** Several configs ramped across runs (343 → 365 → 382) while b12x was still tuning kernels. Single-stream from u20 on uses 3 warm-ups and 3 measured runs.
- **Exact output equality is the wrong test for a cache tier.** At temperature 0, a full prefill and a GPU prefix hit already differ by a word or two because they take different kernel paths. Compare a CPU-tier hit against that baseline difference.

## Files

| Path | What |
|---|---|
| `launch.sh` | Current serving command (`VISION=bf16` or `quant`) |
| `patches/build_patches.sh OUT` | Rebuilds the 13 patched files from the image and the pinned PR diffs; checked byte-identical to the files in service |
| `patches/upstream/` | PR diffs as of 2026-11-21: #68474 (the #68472–#68474 stack) and #66277 |
| `patches/fix*.py` | Hand-applied hunks that don't apply on `8586ccb` (imports and one call site) |
| `patches/local/` | The KV-offload `expandable_segments` exemption |
| `tools/fastprobe.py` | Sanity checks (math, tool call, vision) → single stream → 8-way → cold 60K TTFT; appends to `results.jsonl` |
| `tools/offload_check.py` | Forces a prefix out of the GPU pool and checks it comes back from the CPU tier |
| `tools/mk_bf16vis.py`, `strip_vis_shard.py` | Build a model dir with the BF16 vision tower swapped in (symlinks; source snapshots untouched) |
| `tools/wsize.py` | Checkpoint size by component |
| `results.jsonl` | Every probe run (UTC). `u*`: upstream configs; `c*`: community image; `prod-*`: serving hardening. Runs before 08:16Z used the older 0-warm-up method. |
