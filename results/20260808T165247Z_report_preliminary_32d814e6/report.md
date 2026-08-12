# Jaguar Native Evaluation: Preliminary Report

This report records implemented evidence and current blockers. It is not a performance conclusion and the native baseline is not frozen.

## Platform

- Board: `BirmanPlus-KRK2e` `RevB`
- CPU: Ryzen Embedded V4A46X, 6 cores / 12 threads
- NPU: KRK/XDNA2 via `amdxdna`
- Display: 10 DRM endpoints exposed (including eDP/writeback), 1 physical connector connected
- Network: ax88179_178a over USB 2.0 480 Mb/s bus with 1 GbE PHY

## Idle Baseline

- Samples: 61 over 60.000 seconds
- Aggregate CPU utilization: 0.238%
- GPU busy p99: 0.0%
- APU PPT mean power: 3.196 W
- Power boundary: sampled APU PPT only; no whole-board USB-meter samples or energy counter
- Sensor warning: ACPI is approximately 66 C while AMDGPU is approximately 17 C; mapping is unresolved

## OI CPU Performance

Synthetic input, model-only timing; these figures do not establish segmentation accuracy or end-to-end latency.

- Inferences: 426 over 60.016 seconds
- FPS: 7.098
- Latency p50/p95/p99: 139.008 / 140.916 / 148.423 ms
- APU PPT mean: 22.074 W
- Approximate FPS/APU-PPT-W: 0.322

## STREAM DDR Bandwidth

Each result uses three 610.4 MiB arrays (1.8 GiB total), 20 repetitions, and reports the best validated rate in decimal MB/s.

| Threads | Copy MB/s | Scale MB/s | Add MB/s | Triad MB/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 54999.4 | 38957.6 | 41087.1 | 41223.8 |
| 2 | 75422.3 | 50229.3 | 53421.1 | 53296.3 |
| 4 | 77439.3 | 48950.2 | 51751.9 | 51535.3 |
| 6 | 73357.1 | 45567.8 | 49147.5 | 49021.8 |
| 12 | 70773.1 | 42203.8 | 46852.0 | 46679.0 |

Observed saturation: Copy peaks at four threads; Scale, Add, and Triad peak at two threads. Six and twelve threads regress.

## CoreMark CPU Scaling

Every variant passed both standard seed validations and its timed region exceeded 10 seconds. PPT efficiency is approximate because power is sampled over the whole process, including setup, rather than integrated over only the CoreMark timed region.

| Threads | CoreMark iter/s | CPU util % | APU PPT W | iter/s/PPT-W | ACPI p99 C |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 35463.089 | 8.221 | 7.189 | 4932.841 | 71.0 |
| 6 | 173761.946 | 41.724 | 12.350 | 14070.203 | 74.0 |
| 12 | 265183.603 | 81.958 | 17.271 | 15354.354 | 80.0 |

## Sysbench CPU Scaling

| Threads | Events/s | Avg latency ms | p95 latency ms | Max latency ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4538.08 | 0.22 | 0.22 | 0.58 |
| 6 | 23948.22 | 0.25 | 0.27 | 0.86 |
| 12 | 25049.69 | 0.48 | 0.52 | 12.00 |

## CPU Stress

- stress-ng: 890862 bogo operations over 60.00 seconds
- Real-time rate: 14847.25 bogo ops/s
- Failed stressors: 0

## Dhrystone 2.1

UnixBench Dhrystone C 2.1 register variant with BYTE modifications; compare only with equivalently labeled builds.

- Telemetry-backed single process: 70733882.0 loops/s, 40258.3 DMIPS
- Mean sampled APU PPT: 6.905 W
- Official UnixBench median, one copy: 70800848.0 loops/s (40296.4 DMIPS)
- Official UnixBench median, 12 copies: 512613131.1 loops/s (291754.8 aggregate DMIPS)
- 12-copy scaling: 7.240x; parallel efficiency 60.34%

## Memory Latency

- Standard memcpy: 28105.9 MB/s
- SSE2 non-temporal fill: 63755.0 MB/s
- 64 MiB single random-read extra latency: 111.4 ns without huge pages; 100.5 ns with huge pages

## Vulkan Compute

- Device: AMD Ryzen Embedded V4A46X (RADV GFX1153)
- Driver: radv / Mesa 25.2.8-0ubuntu0.24.04.2
- FP32 scalar: 1428.40 GFLOPS
- FP16 matrix: 4372.52 GFLOPS
- INT8 matrix: 4540.95 GIOPS
- Device-to-device copy: 49.64 GB/s

## Vulkan Graphics

- Upstream vkmark 2025.01 score: 12639 at 1920x1080 via wayland
- Device: AMD Ryzen Embedded V4A46X (RADV GFX1153)
- Mean GPU busy: 98.32%
- Mean sampled APU PPT: 15.533 W
- FPS is render throughput under the Wayland compositor, not physical panel refresh rate.

| Scene | Options | FPS | Frame time ms |
| --- | --- | ---: | ---: |
| vertex | device-local=true | 24442 | 0.041 |
| vertex | device-local=false | 24454 | 0.041 |
| texture | anisotropy=0 | 17257 | 0.058 |
| texture | anisotropy=16 | 16424 | 0.061 |
| shading | shading=gouraud | 14095 | 0.071 |
| shading | shading=blinn-phong-inf | 13502 | 0.074 |
| shading | shading=phong | 12977 | 0.077 |
| shading | shading=cel | 12580 | 0.079 |
| effect2d | kernel=edge | 3246 | 0.308 |
| effect2d | kernel=blur | 1164 | 0.859 |
| desktop | <default> | 4846 | 0.206 |
| cube | <default> | 12094 | 0.083 |
| clear | <default> | 7233 | 0.138 |

## Status

- **audio**: blocked — physical loopback fixture is not confirmed
- **cpu**: ready
- **cpu_counters**: blocked — hardware counters are blocked by perf_event_paranoid=4
- **display_4_to_6**: blocked — only 1 physical display is connected
- **ethernet_usb_baseline**: ready
- **idle_telemetry**: pass
- **inventory**: pass
- **isp**: blocked — no /dev/media* node is present; no /dev/video* node is present
- **media_decode**: ready
- **memory**: ready
- **oi_graph**: ready
- **oi_quantization**: blocked — calibration/evaluation data and accuracy acceptance are missing; xrt-smi is not installed
- **oi_source_integrity**: pass
- **sfp_plus**: blocked — no native PCI network interface is active; compatible DAC/optics and link peer are not confirmed
- **storage**: ready
- **vulkan**: ready

## Next Gates

- Run stressapptest, media, network, and storage benchmark scenarios.
- Run OI accuracy evaluation with production preprocessing and labeled segmentation data.
- Install a KRK-compatible Ryzen AI/VitisAI execution provider and prove NPU operator assignment.
- Connect four and then six independent 1080p60 sinks.
- Provide cameras, SFP+ optics/DAC and peer, audio loopback, and OI calibration/evaluation data.
- Run isolated, normal, peak, thermal, and 24-hour reliability scenarios before freezing the native baseline.
