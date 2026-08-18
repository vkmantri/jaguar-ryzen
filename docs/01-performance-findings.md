# Performance Findings

**Board:** Jaguar P122a devkit, BirmanPlus-KRK2e, BIOS RIJ0071C

**SoC:** AMD Ryzen Embedded V4A46X, 6 cores / 12 threads

**NPU:** 1022:17f0 rev 0x20 → NPU Krackan 2 (npu6), aie2p, 6×8 tile array, firmware 1.1.2.64

**OS:** Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic

**Displays:** 10 DRM endpoints exposed (including eDP/writeback). When it comes to displays, this part is just awesome. You can have up to 10 displays.

---

# CPU & Display Graphics

## Idle Baseline

- Aggregate CPU utilization: 0.238%
- GPU busy: 0.0%
- AMDGPU hwmon mean power: 3.196 W
- Temperature sensors:
  - ACPI: ~66 °C
  - AMDGPU: ~17 °C

## Dhrystone

### Telemetry-backed single process

- 70,733,882.0 loops/s
- 40,258.3 DMIPS

### Mean sampled APU PPT

- 6.905 W

### Official UnixBench median

| Configuration | Result |
|--------------|---------|
| One copy | 70,800,848.0 loops/s (40,296.4 DMIPS) |
| 12 copies | 512,613,131.1 loops/s (291,754.8 aggregate DMIPS) |

- 12-copy scaling: **7.240×**
- Parallel efficiency: **60.34%**

## STREAM DDR Bandwidth

Each result uses three 610.4 MiB arrays (1.8 GiB total), 20 repetitions, and reports the best validated rate in decimal MB/s.

| Threads | Copy MB/s | Scale MB/s | Add MB/s | Triad MB/s |
|----------|----------:|-----------:|---------:|-----------:|
| 1 | 54,999.4 | 38,957.6 | 41,087.1 | 41,223.8 |
| 2 | 75,422.3 | 50,229.3 | 53,421.1 | 53,296.3 |
| 4 | 77,439.3 | 48,950.2 | 51,751.9 | 51,535.3 |
| 6 | 73,357.1 | 45,567.8 | 49,147.5 | 49,021.8 |
| 12 | 70,773.1 | 42,203.8 | 46,852.0 | 46,679.0 |

## CoreMark CPU Scaling

> PPT efficiency is approximate because power is sampled over the whole process, including setup, rather than integrated over only the CoreMark timed region.

| Threads | CoreMark iter/s | CPU util % | APU PPT W |
|-----------|---------------:|-----------:|----------:|
| 1 | 35,463.089 | 8.221 | 7.189 |
| 6 | 173,761.946 | 41.724 | 12.350 |
| 12 | 265,183.603 | 81.958 | 17.271 |

## VkPeak

| Metric | Result |
|----------|---------|
| fp32-scalar | 1428.16 GFLOPS |
| fp32-vec4 | 978.43 GFLOPS |
| fp16-scalar | 1423.27 GFLOPS |
| fp16-vec4 | 2267.00 GFLOPS |
| fp16-matrix | 4371.25 GFLOPS |
| fp64-scalar | 37.51 GFLOPS |
| fp64-vec4 | 37.20 GFLOPS |
| int32-scalar | 240.98 GIOPS |
| int32-vec4 | 238.89 GIOPS |
| int16-scalar | 1277.97 GIOPS |
| int16-vec4 | 2329.20 GIOPS |
| int64-scalar | 80.26 GIOPS |
| int64-vec4 | 66.23 GIOPS |
| int8-dotprod | 4048.03 GIOPS |
| int8-matrix | 4535.94 GIOPS |
| bf16-dotprod | 0.00 GFLOPS |
| bf16-matrix | 0.00 GFLOPS |
| fp8-matrix | 0.00 GFLOPS |
| bf8-matrix | 0.00 GFLOPS |
| copy-h2h | 18.64 GB/s |
| copy-h2d | 15.38 GB/s |
| copy-d2h | 19.00 GB/s |
| copy-d2d | 49.69 GB/s |

## VkMark

```text
=======================================================
    vkmark 2025.01
=======================================================
    Vendor ID:      0x1002
    Device ID:      0x1902
    Device Name:    AMD Ryzen Embedded V4A46X (RADV GFX1153)
    Driver Version: 104865800
    Device UUID:    81083e57af5de00212308be7d6da98cf
=======================================================
[vertex] device-local=true: FPS: 24442 FrameTime: 0.041 ms
[vertex] device-local=false: FPS: 24454 FrameTime: 0.041 ms
[texture] anisotropy=0: FPS: 17257 FrameTime: 0.058 ms
[texture] anisotropy=16: FPS: 16424 FrameTime: 0.061 ms
[shading] shading=gouraud: FPS: 14095 FrameTime: 0.071 ms
[shading] shading=blinn-phong-inf: FPS: 13502 FrameTime: 0.074 ms
[shading] shading=phong: FPS: 12977 FrameTime: 0.077 ms
[shading] shading=cel: FPS: 12580 FrameTime: 0.079 ms
[effect2d] kernel=edge: FPS: 3246 FrameTime: 0.308 ms
[effect2d] kernel=blur: FPS: 1164 FrameTime: 0.859 ms
[desktop] <default>: FPS: 4846 FrameTime: 0.206 ms
[cube] <default>: FPS: 12094 FrameTime: 0.083 ms
[clear] <default>: FPS: 7233 FrameTime: 0.138 ms
=======================================================
                                   vkmark Score: 12639
=======================================================
```

---

# Machine Learning Performance

## Model

**OI**

- Fully convolutional
- 127 nodes
- Opset 17
- Input: `[1,3,576,960]` FP32
- Output: `[1,4,576,960]` FP32

## 1. Headline Numbers

All figures are single-stream, measured over 30-second windows and sampled every 0.5 seconds.

| Configuration | FPS | Latency p50 | Latency p99 | SoC Power | CPU Load |
|---------------|----:|------------:|------------:|----------:|---------:|
| NPU BF16, optimize_level 3 | 16.83 | 57.99 ms | 60.49 ms | 9.46 W | 7.7% |
| NPU INT8 (XINT8), optimize_level 3 | 42.13 | 23.16 ms | 24.76 ms | 14.00 W | 8.7% |

---

## 2. Performance per Watt

### Total SoC Power (AMDGPU hwmon PPT)

CPU FP32 was run to establish a baseline using the same OI model.

| Configuration | FPS | SoC Power | FPS/W |
|--------------|----:|----------:|------:|
| CPU FP32 | 5.69 | 21.28 W | 0.27 |
| NPU BF16 | 16.83 | 9.46 W | 1.78 |
| NPU INT8 | 42.13 | 14.00 W | 3.01 |

**Observations**

- INT8 wins on absolute performance-per-watt.
- BF16 has the lowest marginal power draw, only **1.42 W above idle**.

### Dedicated NPU Power Rail

The amdxdna driver exposes a separate NPU rail:

```text
/sys/class/hwmon/hwmon4/power1_input
```

Label: `NPU_power`

| Configuration | NPU Rail Mean | NPU Rail Max |
|--------------|--------------:|-------------:|
| Idle | 0.000 W | 0.000 W |
| CPU FP32 Run | 0.004 W | 0.090 W |
| NPU BF16 | 0.416 W | 0.691 W |
| NPU INT8 | 1.724 W | 1.800 W |

The NPU rail remains approximately zero during CPU inference, independently verifying that CPU-only runs do not exercise the NPU.

---

## 3. Compiler Settings Dominate BF16 Performance

The largest performance swing measured during evaluation was driven by compiler optimization level rather than precision.

| BF16 Configuration | FPS | p50 Latency | Compile Time |
|------------------|----:|------------:|-------------:|
| optimize_level 1, preferred_data_storage=auto | 2.80 | 352.37 ms | 423 s |
| optimize_level 1, preferred_data_storage=vectorized | 2.80 | 352.33 ms | 425 s |
| optimize_level 3, preferred_data_storage=vectorized | 17.05 | 57.63 ms | 803 s |

**Results**

- optimize_level 1 → 3 yields a **6.1× speedup**
- optimize_level 1 is the documented default
- At optimize_level 1, NPU performance is slower than the CPU (2.80 FPS vs 5.69 FPS)
- At optimize_level 3, the NPU becomes significantly faster

**Tradeoff**

- Optimization level 3 approximately doubles compile time
- Compilation is a one-time cost and the result is cached

---

## 4. What Was Not Measured

- **iGPU inference:** No result. There is currently no ONNX inference path to the iGPU on Linux.
- **Multi-stream or batched throughput:** All tests use batch size 1.
- **Thermal sustainability:** Measurements were limited to 30-second runs.
- **Wall power:** Power measurements come from on-SoC sensors rather than an external meter.

---

## 5. Device-Level NPU Capability

According to `xrt-smi validate` synthetic kernels:

| Test | Result |
|--------|---------|
| GEMM INT8 | 5.7 TOPS |
| Latency | 53.0 µs |
| Throughput | 82,310 op/s |

---

# Language Models

## Phi-3.5-mini-instruct

### 1. Result

The LLM runs on the NPU.

This was verified independently of benchmark output. The driver reported:

- 247 active NPU hardware-context samples
- 883 command submissions

during execution.

### Performance

| Metric | Value |
|----------|---------|
| Prompt processing (TTFT), avg | 237,487 µs (237.5 ms) |
| Prompt processing throughput | 538.98 tokens/s |
| Token generation, avg | 66,883.8 µs/token (14.95 tokens/s) |
| Token generation, p50 | 66,164.4 µs |
| Token sampling | 70.5 µs (14,186 tokens/s) |
| E2E generation loop, avg | 8665.0 ms |
| E2E generation loop, p50/stddev | 8662.1 ms / 14.2 ms |
| Peak working set | 12,713,615,360 B (11.84 GiB) |
| Wall clock (5 repetitions + warmup) | 67.1 s |

### Configuration

- Batch size: 1
- Prompt length: 128 tokens
- Generation length: 128 tokens
- Repetitions: 5
- Warmup runs: 1

### Power

| Rail | Mean | Max |
|--------|-----:|----:|
| SoC Package (AMDGPU PPT) | 15.34 W | 17.03 W |
| NPU Rail (amdxdna NPU_power) | 1.53 W | 1.77 W |

Idle baseline:

- SoC: 8.04 W
- NPU: 0.000 W

LLM inference therefore adds approximately:

- **7.3 W above idle**

(126 samples at 0.5-second intervals)

---

## 2. Comparison with AMD's Documented Expected Output

| Metric | This Board (Krackan) | AMD Example | Ratio |
|----------|-----------------:|-----------:|------:|
| Time to First Token | 237,487 µs | 169,860 µs | 1.40× slower |
| Prompt Processing | 538.98 tok/s | 753.56 tok/s | 0.72× |
| Token Generation | 14.95 tok/s | 49.13 tok/s | 0.30× |
| E2E Generation Loop | 8665.0 ms | 2755.09 ms | 3.15× slower |
| Peak Working Set | 11.84 GiB | 3.30 GiB | 3.59× larger |

AMD's documentation does not state which device produced the reference measurements. Therefore, the gap cannot be attributed to hardware differences with certainty.

### Tuning Attempted: No Effect

Given the substantial impact of compiler optimization in CNN workloads, several tuning options were evaluated.

| Variant | Token Generation | E2E | Peak Working Set |
|----------|----------------:|----:|-----------------:|
| Default | 14.95 tok/s | 8665.0 ms | 12,713,615,360 B |
| `-ml 256` | 14.94 tok/s | 8672.0 ms | 12,718,346,240 B |
| `--reuse_generator` | 14.94 tok/s | 8669.5 ms | 12,731,240,448 B |

All three configurations produced results within 0.1% of each other.

Unlike the CNN workflow, this model arrives pre-quantized and pre-compiled from Hugging Face, leaving no compiler optimization level available to tune.

---

## Most Likely Explanation: Memory Bandwidth

INT4 LLM token generation is generally memory-bandwidth bound because each generated token requires repeated streaming of model weights.

Measured STREAM bandwidth on this system:

| Kernel | Best Rate |
|----------|----------:|
| Copy | 70,773 MB/s |
| Add | 46,852 MB/s |
| Scale | 42,204 MB/s |

If AMD's published results were generated on a Strix Halo platform (~256-bit LPDDR5X, roughly 256 GB/s memory bandwidth), the memory bandwidth ratio would be approximately:

```text
256 / 70.8 ≈ 3.6×
```

Observed token-generation slowdown:

```text
49.13 / 14.95 ≈ 3.29×
```

The two ratios are remarkably close.

This is an inference, not a verified conclusion, because AMD does not disclose the platform used for their reference measurements.

Validating the hypothesis would require executing the identical model on a known Strix or Strix Halo system.

