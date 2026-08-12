# ASR on the Ryzen AI NPU — OpenAI Whisper (Linux)

**Date:** 2026-08-09
**Board:** Jaguar P122a — Ryzen Embedded V4A46X, `NPU Krackan 2` (npu6), `aie2p` 6×8, firmware 1.1.2.64
**Stack:** Ryzen AI 1.8.0 Linux, XRT 2.25.37, amdxdna DKMS 0.15
**Demo:** https://github.com/amd/RyzenAI-SW/blob/main/Demos/ASR/Whisper/README.md
**Model:** `amd/whisper-small-onnx-npu` (1.02 GiB) — see §2 for why not whisper-base

Covers the ASR flow only. The CNN reports (`docs/01`–`03`) and the LLM report
(`docs/llm/01`) are unchanged.

---

## 1. Result — the strongest NPU case measured so far

Audio: `1089-134686-0000.wav`, **10.44 s**, 16 kHz mono (LibriSpeech sample shipped with the demo).

| | TTFT | RTF | Speed vs real-time | SoC power | NPU rail | Wall |
|---|---:|---:|---:|---:|---:|---:|
| **NPU** | **0.05 s** | **0.22** | **4.55× faster** | 9.63 W | 0.087 W | 11.3 s |
| CPU | 0.32 s | 1.21 | **0.83× — slower than real-time** | 17.19 W | 0.000 W | 16.2 s |

**The headline is not the speedup, it is the threshold.** At RTF 1.21 the CPU **cannot keep up
with real-time audio** — a live transcription pipeline would fall progressively behind. At
RTF 0.22 the NPU has 4.5× headroom. This is a capability difference, not a performance
preference.

Both devices produced the **identical** transcription:

> "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat
> mutton pieces to be ladled out in thick, peppered, flour-fattened sauce."

(Correct for this LibriSpeech utterance.)

### Power

Idle baseline on this board is 8.04 W SoC / 0.000 W NPU.

| | SoC mean | Over idle | NPU rail mean |
|---|---:|---:|---:|
| NPU | 9.63 W | **+1.59 W** | 0.087 W |
| CPU | 17.19 W | **+9.15 W** | 0.000 W |

**ASR on the NPU costs 5.8× less incremental power than on the CPU**, while being 5.5× faster
in RTF. The NPU rail reading 0.000 W during the CPU run independently corroborates that the
CPU run did not touch the NPU.

### Verification

NPU residency confirmed from the driver, not from the demo's own output: **12 active hardware
context samples, 37 command submissions** during the run. The CPU run showed no active
contexts, as expected.

This mattered here — see §3.

Runs:
- NPU: `results/20260809T230719Z_asr_whisper_small_npu_9bf2ad6f`
- CPU: `results/20260809T230751Z_asr_whisper_small_cpu_1e941dbf`

---

## 2. The README documents a model the code cannot download

**AMD's README repeatedly instructs you to use `whisper-base`**, and quotes its NPU operator
statistics for exactly that model:

> "#### NPU Run for Whisper-Base — When running inference on the NPU, 100% of the encoder
> operators and 93.4% of the decoder operators are executed on the NPU."

Running it as documented fails:

```
$ python run_whisper.py --model-type whisper-base --device npu --input audio_files/1089-134686-0000.wav
ValueError: Unsupported model_type 'whisper-base' for ONNX auto-download.
```

**Cause.** `whisper-base` and `whisper-tiny` are listed in the script's `--model-type`
`choices`, so argument validation passes — but `download_whisper_onnx()` has no mapping for
them:

```python
hf_model_map = {
    "whisper-small":          "amd/whisper-small-onnx-npu",
    "whisper-medium":         "amd/whisper-medium-onnx-npu",
    "whisper-large-v3-turbo": "amd/whisper-large-turbo-onnx-npu"
}
```

There is no `amd/whisper-base-onnx-npu` repository to fall back to.

**Workaround.** Use `whisper-small` (the smallest model the code can fetch), or supply
`--encoder` / `--decoder` ONNX paths explicitly if you have base models from elsewhere.

Available NPU models:

| Model type | Hugging Face repo | Size |
|---|---|---:|
| `whisper-small` | `amd/whisper-small-onnx-npu` | 1.02 GiB |
| `whisper-medium` | `amd/whisper-medium-onnx-npu` | 4.33 GiB |
| `whisper-large-v3-turbo` | `amd/whisper-large-turbo-onnx-npu` | — |

All ungated. AMD flags `whisper-medium` as needing `--system-stack-size=512` to compile, so
`whisper-small` is also the lower-risk starting point.

---

## 3. The demo emits no NPU evidence of its own

`run_whisper.py` does not raise ONNX Runtime's log severity, so **no VitisAI EP partitioning
markers appear** in its output:

```
$ grep -c "Actually running on NPU" run.log
0
$ grep -c "No. of Operators" run.log
0
```

The run prints `Selected Provider Options: … VitisAIExecutionProvider …`, but that only shows
what was *requested*, not what executed — the same weakness that lets a silent CPU fallback go
unnoticed (see `docs/03-issues-and-workarounds.md`, Problem 1).

Two ways to obtain real evidence:

1. **Driver hardware contexts** (what `scripts/asr/benchmark_asr.py` does):
   ```bash
   xrt-smi examine --report aie-partitions      # status Active, submissions incrementing
   ```
2. Patch `run_whisper.py` to set `session_options.log_severity_level = 1`, which makes the EP
   print its partitioning on a cold compile.

The README's claim of "100% encoder / 93.4% decoder operators on NPU" could **not** be
reproduced from the demo's own output, because the demo does not print it — and in any case
that figure is quoted for `whisper-base`, which cannot be downloaded.

---

## 4. Reproducing

### Setup

```bash
mkdir -p ~/run_asr && cd ~/run_asr
# Demos/ASR/Whisper from github.com/amd/RyzenAI-SW: run_whisper.py, requirements.txt,
# config/, audio_files/
```

**Install dependencies with numpy pinned.** The requirements pull `torch==2.8.0` and
`transformers==4.52.4`; without a constraint, pip may raise numpy above 2.0 and break the
Ryzen AI stack (`flexml` pins `numpy<=1.26.4` — see `docs/03`, Problem 7):

```bash
echo "numpy==1.26.4" > constraints.txt
pip install -c constraints.txt -r requirements.txt \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

Dry-run first (`pip install --dry-run …`) and confirm numpy is unchanged.

> **This install still downgrades Ryzen AI's own pinned packages** and pip reports:
> ```
> voe 1.8.0 requires accelerate==1.12.0, but you have accelerate 1.11.0
> voe 1.8.0 requires transformers==4.57.6, but you have transformers 4.52.4
> torchvision 0.21.0+cpu requires torch==2.6.0, but you have torch 2.8.0+cpu
> ```
> The CNN and LLM flows were re-verified afterwards and both still work (CNN INT8 23.41 ms,
> LLM 14.92 tok/s), but take a `pip freeze` snapshot first so you can roll back.

### Run

```bash
source scripts/ryzenai-env.sh          # the CNN/VitisAI environment, not the LLM one
cd ~/run_asr

# AMD's documented command
python run_whisper.py --model-type whisper-small --device npu \
    --input audio_files/1089-134686-0000.wav

# or with power capture and NPU verification
python scripts/asr/benchmark_asr.py --model-type whisper-small --device npu \
    --input audio_files/1089-134686-0000.wav \
    --results-root results --label asr_whisper_small_npu
```

Note ASR uses `scripts/ryzenai-env.sh` (the VitisAI EP path), **not** `scripts/llm/llm-env.sh` —
Whisper goes through the same VitisAI ONNX Runtime EP as the CNN work, while the LLM flow uses
the separate GenAI `deployment/` tree.

First NPU run compiles encoder and decoder (~270 MB of cache); subsequent runs are warm.

---

## 5. Configuration notes

AMD's shipped `config/vitisai_config_whisper_encoder.json` uses:

```json
"vaiml_config": {
  "optimize_level": 3,
  "fe_experiment": "use-accurate-mode=LayerNorm2PassAdf",
  "aiecompiler_args": "--system-stack-size=512"
}
```

**`optimize_level: 3`** — independent confirmation of the CNN finding that the documented
default of `1` should not be used (see `docs/01-performance-findings.md` §4, where level 1 cost
6.1× on OI). AMD's own demo does not use the default either.

`aiecompiler_args: "--system-stack-size=512"` is not documented in the main Ryzen AI provider
options reference; it appears only in this demo and the README's whisper-medium note.

---

## 6. Limitations

- **Single audio file, single run.** No repetition statistics; RTF and TTFT are from one pass
  over a 10.44 s sample. A second, longer sample (`61-52s.wav`, 43.00 s) ships with the demo
  and has not been measured.
- **No WER/CER measured.** The demo supports `--eval-dir` against LibriSpeech samples; that
  evaluation has not been run. Accuracy is asserted only by the transcription matching between
  CPU and NPU on one utterance.
- **whisper-base not evaluated** — it cannot be downloaded (§2), so the README's operator-split
  figures remain unverified.

## 7. Microphone input is broken as shipped

`--input mic` fails immediately:

```
NameError: name 'sd' is not defined
```

`import sounddevice as sd` is at line 357 **inside `main()`**, so it is function-local, while
`mic_stream()` and its inner `feeder()` are module-level and reference a global `sd` that never
exists. Moving the import to module scope fixes it.

Two further problems for live use:

- **`libportaudio2` is not installed** by `requirements.txt` even though `sounddevice` is listed.
  `sudo apt install libportaudio2`.
- **`silence_duration` is hard-coded to 5.0 s** and the run exits on the first 5 s of quiet.
  Since model load takes ~7 s, the demo often ends before the speaker can react.

A patched copy and a live voice agent built on it are documented in
`docs/voice-agent/01-voice-agent-findings.md`; the unmodified original is preserved at
the original AMD `run_whisper.py`, which is not redistributed here. Obtain it
from AMD's Ryzen AI ASR example.

---

## 7. Artefacts

| Item | Path |
|---|---|
| NPU run (metrics, power, verification) | `results/20260809T230719Z_asr_whisper_small_npu_9bf2ad6f/asr_benchmark.json` |
| CPU run | `results/20260809T230751Z_asr_whisper_small_cpu_1e941dbf/asr_benchmark.json` |
| Benchmark wrapper | `scripts/asr/benchmark_asr.py` |
| Environment | `scripts/ryzenai-env.sh` (shared with the CNN flow) |
