# Result Evidence

This directory publishes compact, structured evidence used by the findings:

- CPU: CoreMark, Dhrystone, sysbench, and stress-ng
- Memory: STREAM and tinymembench
- GPU: vkpeak and upstream vkmark
- AI: OI CPU/NPU runs, reference metadata, and XRT validation
- Applications: Phi-3.5 LLM and Whisper ASR CPU/NPU benchmarks
- Platform: inventory, idle baseline, readiness, and the consolidated report

Raw telemetry streams, generated tensors, model binaries, compiler logs, and
failed exploratory runs are excluded from the repository. Power figures are
sampled APU PPT unless a result explicitly states another boundary.