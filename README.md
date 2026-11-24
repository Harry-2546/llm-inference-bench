# llm-inference-bench

Reproducible benchmark harnesses and measurement discipline for multi-GPU LLM serving — developed on a 4×RTX 5090 rig (PEX88096 switch, vLLM TP4 (+EP on r10), Qwen3.8-Flash-Next NVFP4 / FP4-Local, MTP speculative decoding; see companion repo [`4x5090-rig`](https://github.com/Harry-2546/4x5090-rig)).

These are the tools behind every performance number I publish. The point of this repo is less the scripts and more the **rules they enforce**.

## Methodology rules (learned the hard way)

1. **Compare step rates, not tok/s, across engines/configs.** `tok/s = steps/s × (1 + draft × accept)` — acceptance rate is output-distribution-dependent and not comparable.
2. **Same tokenizer, same prompt text, temperature stated.** Prompt-token counts must match across engines before any ratio is meaningful.
3. **Exclude the first run.** Cold-start effects (autotune, expert-cache warm-up) silently reverse conclusions; one of our published comparisons was an artifact of test order (first-run penalty always landing on the same arm).
4. **Decompose public benchmark scores before quoting them.** A community headline of "549 tok/s" turned out to be ~76% JSON-copy tasks. We report composition.
5. **Pin the noise floor first.** Identical-prompt accept-rate runs spanned 0.45–0.74; anything inside that band is not an effect.
6. **Locked-clock sweeps to attribute bottlenecks**, e.g. step time `T ≈ 19.5/f(GHz) + 1.8 ms` → single-stream decode was ~78% core-frequency-bound; PCIe uplink averaged only ~1.3 GB/s per card.
7. **Measure the noise before using a small threshold.** The same config measured twice gave 8-way results 4% apart; 1% decisions need alternated repeats, not one run per arm.
8. **Don't test a cache tier for exact output equality.** At temperature 0, a full prefill and a GPU prefix hit already differ by a word or two (different kernel paths); judge a CPU-tier hit against that baseline difference.

## Contents

| Path | What |
|---|---|
| `pc2/bench/bench.py`, `ab.py`, `accept.py` | Throughput / A-B config comparison / speculative-decoding acceptance harnesses (OpenAI-compatible API) |
| `pc2/bench/coresweep.*`, `conc_strata.py`, `bench_strata_r10.py` | Clock-locked bottleneck sweeps; concurrency curves; cross-engine (Strata vs vLLM) comparisons |
| `pc2/bench/bt/cfg*.json` | Basalt C++ engine configs (2-GPU partitions) |
| `pc2/bench/*.log` | Raw run logs, untouched (the mess is the evidence) |
| `pc2/scripts/pc2speed` | Live serving dashboard: single/concurrent/prefill/acceptance + 24 h recompute-vs-total-time ratio (prefix-cache thrash visibility) |
| `pc2/fast-2026-11/` | **FP4-Local speed-up (2026-11-21)**: per-change A/B table, current launch script, reproducible patch build (3 unmerged upstream PRs + a KV-offload fix), probe/offload-check tools, all raw results |
| `pc2/launch-r10.sh` | Earlier NVFP4 launch config on the r10 community image (YaRN 524K corrected, `--max-num-seqs` matched to CUDA-graph capture boundary) |
| `strata/` | Cross-rig comparison records |

## Headline results (all reproducible with these scripts)

**Current (2026-11-21, FP4-Local on upstream vLLM `8586ccb` + patches; details in [`pc2/fast-2026-11/`](pc2/fast-2026-11/)):**

| Metric | Value |
|---|---|
| Decode, single stream (think off) | 379 tok/s (+11.6% vs our NVFP4 baseline of 340) |
| Aggregate, 8-way | 1,375 tok/s (+3.6% vs 1,327) |
| Cold 60K-token prompt TTFT | 4.65 s (−32% from a 4096-token prefill chunk) |
| GSM8K 250 / GPQA 198 | 96.8% / 92.42% |
| CPU KV tier (56 GB) hit | 52,800-token prefix: TTFT 4.3 s → 0.45–1.9 s |

**Earlier (r10 community image, NVFP4):**

| Metric | Value |
|---|---|
| Step rate | 112.3 idle → 116.1 with VRAM +4000 offset (recipe's published baseline 106) |
| Decode, single stream (temp 0 + thinking) | 292 code / 256 prose / 265 zh tok/s |
| Aggregate, 8-way | ~1,200 tok/s |
| Prefill | ~12.5 K tok/s |
| `--max-num-seqs` fix | 16-way: 552 → 1,136 tok/s (CUDA-graph capture boundary) |
| vLLM vs C++ engine | 1.5× single-stream gap lives in per-step latency (5.9 vs 9.1 ms), not acceptance rate |

