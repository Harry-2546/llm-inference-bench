# llm-inference-bench

Reproducible benchmark harnesses and measurement discipline for multi-GPU LLM serving — developed on a 4×RTX 5090 rig (PEX88096 switch, vLLM TP4+EP, Qwen3.8-Flash-Next NVFP4, MTP speculative decoding; see companion repo [`4x5090-rig`](https://github.com/Harry-2546/4x5090-rig)).

These are the tools behind every performance number I publish. The point of this repo is less the scripts and more the **rules they enforce**.

## Methodology rules (learned the hard way)

1. **Compare step rates, not tok/s, across engines/configs.** `tok/s = steps/s × (1 + draft × accept)` — acceptance rate is output-distribution-dependent and not comparable.
2. **Same tokenizer, same prompt text, temperature stated.** Prompt-token counts must match across engines before any ratio is meaningful.
3. **Exclude the first run.** Cold-start effects (autotune, expert-cache warm-up) silently reverse conclusions; one of our published comparisons was an artifact of test order (first-run penalty always landing on the same arm).
4. **Decompose public benchmark scores before quoting them.** A community headline of "549 tok/s" turned out to be ~76% JSON-copy tasks. We report composition.
5. **Pin the noise floor first.** Identical-prompt accept-rate runs spanned 0.45–0.74; anything inside that band is not an effect.
6. **Locked-clock sweeps to attribute bottlenecks**, e.g. step time `T ≈ 19.5/f(GHz) + 1.8 ms` → single-stream decode was ~78% core-frequency-bound; PCIe uplink averaged only ~1.3 GB/s per card.

## Contents

| Path | What |
|---|---|
| `pc2/bench/bench.py`, `ab.py`, `accept.py` | Throughput / A-B config comparison / speculative-decoding acceptance harnesses (OpenAI-compatible API) |
| `pc2/bench/coresweep.*`, `conc_strata.py`, `bench_strata_r10.py` | Clock-locked bottleneck sweeps; concurrency curves; cross-engine (Strata vs vLLM) comparisons |
| `pc2/bench/bt/cfg*.json` | Basalt C++ engine configs (2-GPU partitions) |
| `pc2/bench/*.log` | Raw run logs, untouched (the mess is the evidence) |
| `pc2/scripts/pc2speed` | Live serving dashboard: single/concurrent/prefill/acceptance + 24 h recompute-vs-total-time ratio (prefix-cache thrash visibility) |
| `pc2/launch.sh.current` | Our production vLLM launch config (YaRN 524K corrected, `--max-num-seqs` matched to CUDA-graph capture boundary) |
| `strata/` | Cross-rig comparison records |

## Headline results (all reproducible with these scripts)

| Metric | Value |
|---|---|
| Step rate | 112.3 idle → 116.1 with VRAM +4000 offset (recipe's published baseline 106) |
| Decode, single stream (temp 0 + thinking) | 292 code / 256 prose / 265 zh tok/s |
| Aggregate, 8-way | ~1,200 tok/s |
| Prefill | ~12.5 K tok/s |
| `--max-num-seqs` fix | 16-way: 552 → 1,136 tok/s (CUDA-graph capture boundary) |
| vLLM vs C++ engine | 1.5× single-stream gap lives in per-step latency (5.9 vs 9.1 ms), not acceptance rate |

