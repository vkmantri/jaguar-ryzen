# Jaguar Ryzen Evaluation

Some early work on development platform with a Ryzen Embedded V4A46X, RDNA graphics, and a
KRK/XDNA2 NPU.

## Documentation

- [Performance findings](docs/01-performance-findings.md)
- [Ryzen AI workflow guide](docs/02-ryzen-ai-workflow-guide.md)
- [Issues and workarounds](docs/03-issues-and-workarounds.md)
- [Whisper ASR findings](docs/asr/01-asr-whisper-findings.md)
- [Phi-3.5 LLM findings](docs/llm/01-llm-phi35-findings.md)
- [Voice-agent findings](docs/voice-agent/01-voice-agent-findings.md)

## Application Code

- `scripts/voice-agent/`: microphone-to-ASR-to-LLM-to-TTS voice assistant
- `scripts/asr/`: Whisper CPU/NPU benchmark and combined runtime environment
- `scripts/llm/`: Phi-3.5 OGA benchmark and runtime environment
- `scripts/ryzenai-env.sh`: shared Ryzen AI/VitisAI environment

The voice agent expects AMD's `run_whisper.py` and downloaded model/runtime
assets to be installed separately. Those vendor and model files are not
redistributed here.

## Results

`results/` contains compact structured JSON and Markdown evidence referenced by
the findings. Raw tensor dumps, telemetry streams, compiler logs, downloaded
models, vendor packages, and external benchmark sources are intentionally
excluded.

CPU and memory benchmark implementation code and the general profiling harness
are also excluded; only their result artifacts and findings are published.

## Scope

No board user-guide PDFs, customer ONNX models, quantized model binaries,
private keys, credentials, or third-party binary packages are tracked.
