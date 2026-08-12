# LLM on the Ryzen AI NPU — Phi-3.5-mini-instruct (Linux)

**Date:** 2026-08-09
**Board:** Jaguar P122a — Ryzen Embedded V4A46X, `NPU Krackan 2` (npu6), `aie2p` 6×8, firmware 1.1.2.64
**Stack:** Ryzen AI 1.8.0 Linux, XRT 2.25.37, amdxdna DKMS 0.15
**Model:** `amd/Phi-3.5-mini-instruct_rai_1.8.0_npu_4K` (prequantized OGA model, 5.04 GiB, 76 files)
**Procedure:** https://ryzenai.docs.amd.com/en/latest/llm_linux.html

This document covers the LLM (OGA) flow only. It is deliberately separate from the CNN reports
in `docs/01`–`03`, which are unchanged.

---

## 1. Result

The LLM runs on the NPU. Verified independently of the benchmark's own output: the driver
reported **247 active NPU hardware-context samples** with **883 command submissions** during
the run.

Run: `results/20260809T181017Z_llm_phi35_npu_d7646bcc`

| Metric | Value |
|---|---:|
| Prompt processing (time to first token), avg | 237,487 µs (237.5 ms) |
| Prompt processing throughput | 538.98 tokens/s |
| **Token generation, avg** | **66,883.8 µs/token (14.95 tokens/s)** |
| Token generation, p50 | 66,164.4 µs |
| Token sampling | 70.5 µs (14,186 tokens/s) |
| E2E generation loop, avg | 8665.0 ms |
| E2E, p50 / stddev | 8662.1 ms / 14.2 ms |
| Peak working set | 12,713,615,360 B (11.84 GiB) |
| Wall clock (5 repetitions + warmup) | 67.1 s |

Configuration: batch 1, prompt 128 tokens, generate 128 tokens, 5 repetitions, 1 warmup.

### Power

| Rail | Mean | Max |
|---|---:|---:|
| SoC package (`amdgpu` PPT) | **15.34 W** | 17.03 W |
| NPU rail (`amdxdna` NPU_power) | **1.53 W** | 1.77 W |

Idle baseline on this board is 8.04 W SoC / 0.000 W NPU, so LLM inference costs roughly
**7.3 W over idle**. 126 samples at 0.5 s.

Derived: at 14.95 tokens/s and 15.34 W, that is **1.03 J per generated token**
(0.97 tokens/joule) at the package level.

---

## 2. Comparison with AMD's documented expected output

AMD's page shows expected output for the same model. Ours is materially slower:

| Metric | This board (Krackan) | AMD's documented example | Ratio |
|---|---:|---:|---:|
| Time to first token | 237,487 µs | 169,860 µs | 1.40× slower |
| Prompt processing | 538.98 tok/s | 753.56 tok/s | 0.72× |
| **Token generation** | **14.95 tok/s** | **49.13 tok/s** | **0.30×** |
| E2E generation loop | 8665.0 ms | 2755.09 ms | 3.15× slower |
| Peak working set | 11.84 GiB | 3.30 GiB | 3.59× larger |

**AMD's documentation does not state which device produced its numbers.** The gap is
therefore not directly attributable without that information.

### Tuning attempted — no effect

Given that `optimize_level` produced a 6.1× swing in the CNN work, the obvious knobs were
tested. Neither changed anything:

| Variant | Token generation | E2E | Peak working set |
|---|---:|---:|---:|
| Default | 14.95 tok/s | 8665.0 ms | 12,713,615,360 B |
| `-ml 256` (explicit max_length) | 14.94 tok/s | 8672.0 ms | 12,718,346,240 B |
| `--reuse_generator` | 14.94 tok/s | 8669.5 ms | 12,731,240,448 B |

All three agree to within 0.1 %. Unlike the CNN flow, this is not a configuration problem —
the model arrives prequantized and precompiled from Hugging Face, so there is no compiler
optimisation level to raise.

### Most likely explanation: memory bandwidth

INT4 LLM token generation is memory-bandwidth-bound — each token requires streaming the
weights. Measured STREAM bandwidth on this board:

| Kernel | Best rate |
|---|---:|
| copy | 70,773 MB/s |
| add | 46,852 MB/s |
| scale | 42,204 MB/s |

If AMD's figures came from Strix Halo (256-bit LPDDR5X, ~256 GB/s class), the bandwidth ratio
would be ≈3.6×, against the observed 3.29× token-generation gap. That is a close match and
consistent with a bandwidth-bound workload.

**This is inference, not a verified fact** — AMD does not publish the device used. It should
be treated as the leading hypothesis, not a conclusion. Confirming it would require running
the identical model on a known Strix/Strix Halo part.

Note that prompt processing (compute-bound, batched) degrades far less (0.72×) than token
generation (0.30×), which is the signature one would expect if bandwidth rather than compute
were the limit.

### The working-set discrepancy is unexplained

11.84 GiB versus AMD's 3.30 GiB is a 3.59× difference and is **not** explained by the
bandwidth hypothesis. Note the model's `genai_config.json` declares
`context_length: 131072` (128K) despite the model being named `_4K`. Overriding `max_length`
explicitly did not reduce the working set, so the allocation appears independent of the
requested sequence length. This remains an open question.

---

## 3. Reproducing

### Prerequisites

- Ryzen AI 1.8.0 installed and working (see `docs/02-ryzen-ai-workflow-guide.md`)
- ~5.1 GiB disk for the model, ~12 GiB RAM for the working set

**`git-lfs` is not required.** AMD's instructions use `git lfs clone`, but the model repo is
public and ungated, so it can be fetched over plain HTTPS:

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="amd/Phi-3.5-mini-instruct_rai_1.8.0_npu_4K",
                  local_dir="Phi-3.5-mini-instruct_rai_1.8.0_npu_4K", max_workers=8)
PY
```

Took 1 m 46 s for 5.04 GiB.

### Stage the runtime files

```bash
mkdir -p ~/run_llm && cd ~/run_llm
cp -r  $RYZEN_AI_INSTALLATION_PATH/deployment .
cp     $RYZEN_AI_INSTALLATION_PATH/LLM/examples/model_benchmark .
cp     $RYZEN_AI_INSTALLATION_PATH/LLM/examples/amd_genai_prompt.txt .
chmod +x model_benchmark
```

### Environment

```bash
source scripts/llm/llm-env.sh      # from this repo; must be sourced from ~/run_llm
```

Which sets, per AMD's instructions:

```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$PWD/deployment/lib:$LD_LIBRARY_PATH
export RYZENAI_EP_PATH=$PWD/deployment/lib/libonnxruntime_providers_ryzenai.so
```

### Run with power capture and NPU verification

```bash
python scripts/llm/benchmark_llm.py \
  --model Phi-3.5-mini-instruct_rai_1.8.0_npu_4K/ \
  --prompt-length 128 \
  --results-root results --label llm_phi35_npu
```

Or AMD's bare command:

```bash
./model_benchmark -i Phi-3.5-mini-instruct_rai_1.8.0_npu_4K/ -l 128
```

---

## 4. Notes and gotchas specific to the LLM flow

**`model_benchmark` has no NPU execution-provider flag.** `--execution_provider` accepts only
`cpu, cuda, dml, NvTensorRtRtx`. The NPU is selected via `RYZENAI_EP_PATH` and the model's
`genai_config.json` (`provider_options: [{"RyzenAI": {"hybrid_opt_token_backend": "npu"}}]`).
There is therefore **no flag to confirm NPU use** — it must be verified out-of-band:

```bash
xrt-smi examine --report aie-partitions     # status Active, submissions incrementing
```

`scripts/llm/benchmark_llm.py` does this automatically and records the result.

**Missing shared library on first run.** Running `./model_benchmark --help` before setting
`LD_LIBRARY_PATH` fails with
`error while loading shared libraries: libonnxruntime-genai.so`. Source the environment first.

**Memlock.** AMD's page warns that some models need `memlock unlimited`. This board's default
is 4,024,488 KB (3.84 GiB) and the run succeeded without changing it, despite a 11.84 GiB peak
working set — working set is not locked memory. If a model does fail, the documented fix is:

```bash
sudo tee /etc/security/limits.d/99-memlock.conf >/dev/null <<'EOF'
*    soft    memlock    unlimited
*    hard    memlock    unlimited
EOF
```
(then log out and back in).

**Linux does not support the hybrid flow** — NPU-only, per AMD's page. Model generation
(`model_generate`) is also unsupported in this release on Linux; use the prequantized Hugging
Face models.

---

## 5. Artefacts

| Item | Path |
|---|---|
| Benchmark run (metrics, power, NPU verification) | `results/20260809T181017Z_llm_phi35_npu_d7646bcc/llm_benchmark.json` |
| Raw `model_benchmark` output | `results/20260809T181017Z_llm_phi35_npu_d7646bcc/model_benchmark.log` |
| Environment script | `scripts/llm/llm-env.sh` |
| Benchmark wrapper | `scripts/llm/benchmark_llm.py` |
| Cached AMD instructions | `vendor-docs/ryzenai/llm_linux.txt` |
