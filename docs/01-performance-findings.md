# OI on AMD Ryzen Embedded V4A46X (Jaguar P122a) — Performance Findings

**Date:** 2026-08-09
**Board:** Jaguar P122a devkit, `BirmanPlus-KRK2e`, BIOS RIJ0071C
**SoC:** AMD Ryzen Embedded V4A46X, 6 cores / 12 threads
**NPU:** `1022:17f0` rev `0x20` → `NPU Krackan 2` (npu6), `aie2p`, 6×8 tile array, firmware 1.1.2.64
**OS:** Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic
**Model:** OI — fully convolutional, 127 nodes, opset 17, input `[1,3,576,960]` fp32 → output `[1,4,576,960]` fp32

---

## 1. Headline numbers

All figures are single-stream, 30-second measured windows, telemetry sampled at 0.5 s.
Every NPU run is **verified** to have executed on the NPU (see §5).

| Configuration | FPS | Latency p50 | Latency p99 | SoC power | CPU load |
|---|---:|---:|---:|---:|---:|
| CPU fp32, 12 threads | 5.69 | 168.80 ms | 222.11 ms | 21.28 W | 99.7 % |
| NPU BF16, `optimize_level 3` | 16.83 | 57.99 ms | 60.49 ms | 9.46 W | 7.7 % |
| **NPU INT8 (XINT8), `opt_level 3`** | **42.13** | **23.16 ms** | **24.76 ms** | **14.00 W** | **8.7 %** |

**Relative to the CPU baseline:**

| | Throughput | Latency | SoC power |
|---|---:|---:|---:|
| NPU BF16 | **2.96×** faster | 2.91× lower | 0.44× |
| NPU INT8 | **7.40×** faster | 7.29× lower | 0.66× |

---

## 2. Performance per watt

Two framings, because they answer different questions.

**Total SoC power** (`amdgpu` hwmon `PPT`, whole-package) — what the system actually draws:

| Configuration | FPS | SoC power | FPS per watt | Energy per frame |
|---|---:|---:|---:|---:|
| CPU fp32 | 5.69 | 21.28 W | 0.27 | 3737 mJ |
| NPU BF16 | 16.83 | 9.46 W | 1.78 | 562 mJ |
| **NPU INT8** | 42.13 | 14.00 W | **3.01** | **332 mJ** |

**Incremental over idle** (idle PPT = 8.04 W) — the marginal cost of the inference itself:

| Configuration | Incremental power | FPS per incremental watt |
|---|---:|---:|
| CPU fp32 | 13.24 W | 0.43 |
| **NPU BF16** | **1.42 W** | **11.82** |
| NPU INT8 | 5.96 W | 7.07 |

**Energy per frame is the clearest single metric: INT8 on the NPU uses 332 mJ/frame versus
3737 mJ/frame on the CPU — an 11.3× reduction.**

Note the two framings disagree on the winner. INT8 wins on absolute performance-per-watt,
but **BF16 has the lowest marginal power draw of any option** (1.42 W over idle). For a
thermally or battery constrained deployment where OI need only hit ~16 FPS, BF16 is the more
efficient choice despite lower throughput.

### Dedicated NPU power rail

The `amdxdna` driver exposes a separate NPU rail (`/sys/class/hwmon/hwmon4/power1_input`,
label `NPU_power`), world-readable:

| Configuration | NPU rail mean | NPU rail max |
|---|---:|---:|
| Idle | 0.000 W | 0.000 W |
| CPU fp32 run | 0.004 W | 0.090 W |
| NPU BF16 | 0.416 W | 0.691 W |
| NPU INT8 | 1.724 W | 1.800 W |

The NPU rail reads ~0 W during CPU inference, which independently corroborates that the CPU
runs were genuinely not touching the NPU, and vice versa.

---

## 3. The CPU offload argument

The throughput numbers understate the benefit. During CPU inference the SoC sits at
**99.7 % CPU utilisation across all 12 threads** — the machine has nothing left for
application work. NPU inference runs at **7.7–8.7 % CPU**.

For an embedded workload where OI is one stage in a larger pipeline, this matters more than
the raw multiplier: the NPU path leaves essentially the entire CPU free.

The NPU is also markedly more deterministic:

| Configuration | p50 | p99 | max | spread (max−min) |
|---|---:|---:|---:|---:|
| CPU fp32 | 168.80 ms | 222.11 ms | 285.20 ms | **128.68 ms** |
| NPU BF16 | 57.99 ms | 60.49 ms | 61.50 ms | 5.55 ms |
| NPU INT8 | 23.16 ms | 24.76 ms | 26.12 ms | **4.00 ms** |

The CPU's p99 is 32 % above its median and its worst case is 69 % above. The NPU's p99 is
within 7 % of median. For a real-time pipeline with a frame deadline, the NPU's tail
behaviour is the more valuable property.

---

## 4. Compiler settings dominate BF16 performance

The single largest effect measured in this evaluation was **not** precision — it was the
compiler optimisation level.

| BF16 configuration | FPS | p50 | Compile time |
|---|---:|---:|---:|
| `optimize_level 1`, `preferred_data_storage auto` | 2.80 | 352.37 ms | 423 s |
| `optimize_level 1`, `preferred_data_storage vectorized` | 2.80 | 352.33 ms | 425 s |
| **`optimize_level 3`, `preferred_data_storage vectorized`** | **17.05** | **57.63 ms** | 803 s |

- `optimize_level 1` → `3` is a **6.1× speedup**.
- `optimize_level 1` is the **documented default**. With it, the NPU is *slower than the CPU*
  (2.80 vs 5.69 FPS). At level 3 it is 3× faster. Anyone benchmarking with defaults would
  reach the opposite conclusion about this hardware.
- `preferred_data_storage` made **no measurable difference** (352.37 vs 352.33 ms), despite
  the documentation recommending `vectorized` for CNNs.
- Cost: opt3 roughly doubles compile time. This is one-time and cached.

For INT8 the same knob is nearly irrelevant: `opt_level` 0 → 3 moved FPS from 40.47 to 41.84
(+3 %). **The two precisions have completely different tuning sensitivity.**

---

## 5. Execution verification

Every NPU figure in this report is backed by evidence that the work actually ran on the NPU.
This mattered: an early run printed `Test Finished` with exit code 0 while having silently
fallen back to CPU (see the issues report, Problem 1).

Two independent mechanisms:

**Compiler partitioning report** (cold compile only):

| Model | NPU operators | CPU operators | Subgraphs |
|---|---:|---:|---:|
| OI BF16 | `VAIML` 122 | 0 | 1 |
| OI INT8 | `NPU` 484 | `VITIS_EP_CPU` 2 | — |

The INT8 model's 2 CPU nodes are `QuantizeLinear` and `DequantizeLinear` at the graph
boundary — input fp32→int8 and output int8→fp32 conversion. Confirmed via the operator
assignment report. Every interior operator is on the NPU. The node count differs (486 vs 127)
because QDQ pairs inflate the quantized graph.

**Driver hardware context** (works with a warm cache, where nothing is compiled and no log
markers are emitted):

```
xrt-smi examine --report aie-partitions
  pid 103599, context_id 1, status Active, columns [0..7]
  command_submissions / command_completions incrementing
```

The 30-second power runs above were all verified this way.

---

## 6. Accuracy status — read before quoting these numbers

- **BF16: no accuracy caveat from the conversion process.** FP32→BF16 is performed by the
  compiler with no calibration data required.
- **INT8: latency and FPS are valid; accuracy is NOT established.** `oi_data.npz` contains a
  **single image**. INT8 post-training quantization derives activation ranges from calibration
  data, and one sample is not representative. This is recorded in
  `models/oi/int8/oi_xint8.metadata.json` as `accuracy_valid: false`.
- Numerical output has not been compared against a reference for either precision. A CPU fp32
  golden reference exists at `results/20260809T004132Z_oi_reference_0f02baa9/` (verified
  deterministic: two runs agree to 0.0 absolute difference), so this comparison is ready to run.
- The preprocessing normalisation used by OI at training time is **unknown**; the benchmark
  uses an explicit, recorded scheme. This does not affect timing.

**To make INT8 accuracy meaningful, a representative calibration set (typically 100–1000
images) is required.**

---

## 7. What was not measured

- **iGPU: no result.** There is no ONNX inference path to the iGPU on Linux. See the issues
  report for the full explanation and evidence.
- **Multi-stream / batched throughput.** All figures are single-stream, batch 1.
- **Thermal sustained behaviour.** Runs were 30 s; no soak test was performed.
- **Wall power.** All power figures are from on-die sensors, not an external meter.

---

## 8. Reproducing

```bash
source scripts/ryzenai-env.sh
export PYTHONPATH=$PWD/src

# INT8
python -m jaguar_eval model-benchmark --model models/oi/int8/oi_xint8.onnx \
  --provider vitisai --vai-target X2 --vai-opt-level 3 \
  --cache-dir /home/amd/ryzenai/vaip_cache_int8_opt3 --cache-key oi_int8 \
  --warmup 10 --iterations 0 --minimum-duration 30 --telemetry-interval 0.5

# BF16
python -m jaguar_eval model-benchmark --model models/oi/2c07ee086b5450aa/oi.onnx \
  --provider vitisai --vai-config config/vai_ep_bf16_vectorized_opt3.json \
  --cache-dir /home/amd/ryzenai/vaip_cache_vectorized_opt3 --cache-key oi_bf16 \
  --warmup 10 --iterations 0 --minimum-duration 30 --telemetry-interval 0.5

# CPU baseline
python -m jaguar_eval model-benchmark --model models/oi/2c07ee086b5450aa/oi.onnx \
  --provider cpu --warmup 5 --iterations 0 --minimum-duration 30 --telemetry-interval 0.5
```

A run fails loudly rather than reporting CPU numbers as NPU results; pass
`--allow-cpu-fallback` only if you deliberately want the unverified behaviour.

### Source runs for every figure

| Figure set | Run directory |
|---|---|
| CPU fp32 + power | `results/20260809T174053Z_oi_cpu_8f1f73aa` |
| NPU BF16 opt3 + power | `results/20260809T174022Z_oi_vitisai_02690af0` |
| NPU INT8 opt3 + power | `results/20260809T173951Z_oi_vitisai_8ad1544f` |
| Idle power baseline | `results/20260809T174211Z_telemetry_idle_0083aca1` |
| BF16 opt1 (auto) | `results/20260809T061105Z_oi_vitisai_886fa797` |
| BF16 opt1 (vectorized) | `results/20260809T062117Z_oi_vitisai_7193284d` |
| BF16 opt3 (100 iterations) | `results/20260809T062859Z_oi_vitisai_f57d2fa8` |
| INT8 opt0 (100 iterations) | `results/20260809T165937Z_oi_vitisai_477a26a0` |
| INT8 opt3 (100 iterations) | `results/20260809T165948Z_oi_vitisai_9a100529` |
| CPU fp32 golden output | `results/20260809T004132Z_oi_reference_0f02baa9` |
| NPU hardware validation (`xrt-smi`) | `results/20260808T182702Z_npu_xrt-validate_008dbbcc` |

---

## 9. Device-level NPU capability

For context, `xrt-smi validate` synthetic kernels on this NPU:

| Test | Result |
|---|---|
| GEMM INT8 | 5.7 TOPS |
| Latency | 53.0 µs |
| Throughput | 82,310 op/s |

The driver also reports a live `gops` figure per hardware context (84 GOPS observed during
INT8 OI inference). Note this is far below the 5.7 TOPS synthetic peak — OI at 576×960 is not
a GEMM-shaped workload, and this gap is the main indicator of remaining headroom.
