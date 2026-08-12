# Local Voice Agent on the Ryzen AI NPU — metrics and issues

**Date:** 2026-08-10
**Board:** Jaguar P122a — Ryzen Embedded V4A46X, `NPU Krackan 2` (npu6), `aie2p` 6×8
**Stack:** Ryzen AI 1.8.0 Linux, XRT 2.25.37, amdxdna DKMS 0.15
**Audio:** Jabra SPEAK 510 (USB), PipeWire 1.0.5

An end-to-end voice agent running **entirely on-device**: no network, no cloud API.
Built on top of the ASR and LLM work in `docs/asr/01` and `docs/llm/01`, which are unchanged.

```
Jabra mic → Whisper-small (NPU) → text → Phi-3.5-mini (NPU) → reply → Piper TTS → Jabra speaker
```

---

## 1. Measured latency

17 spoken turns, warm models, `--silence-seconds 1.0`, `--max-new-tokens 40`.

| Stage | min | median | max |
|---|---:|---:|---:|
| Whisper ASR | 0.19 s | **0.52 s** | 0.70 s |
| ASR real-time factor | 0.04 | **0.12** | 0.19 |
| LLM generation | 0.22 s | **0.77 s** | 1.76 s |
| LLM tokens produced | 3 | 10 | 22 |
| LLM token rate | 4.0 tok/s | **14.4 tok/s** | 15.0 tok/s |
| **End of speech → first audio out** | **0.69 s** | **1.19 s** | 2.51 s |

Plus a fixed **1.0 s** silence-detection wait before processing begins, included in the
figures above.

Representative turns:

| Question | Reply | First audio |
|---|---|---:|
| "What is the capital of France?" | "Paris." | 0.69 s |
| "What is the capital of Canada?" | "Ottawa" | 0.73 s |
| "Thank you." | "You're welcome." | 0.88 s |
| "What type of model are you?" | "I am Phi, a Microsoft language model." | 1.19 s |
| "Who makes the best cars?" | "Tesla, Toyota, and BMW are often cited…" | 1.93 s |

**Median 1.19 s from finishing a sentence to hearing the reply begin**, fully local.

### How it got there

Initial implementation measured **~7.4 s**. Three changes, in order of impact:

| Change | Saving |
|---|---:|
| Stream sentence chunks and speak the first while the rest generates | ~2.9 s |
| Silence cutoff 2.5 s → 1.0 s | 1.5 s |
| System prompt forcing one short sentence (50-token rambles → 3–22 tokens) | ~1.3 s |

The streaming change matters most: the LLM emits each finished sentence as
`{"chunk": ...}` and playback starts immediately, so generation time is largely hidden behind
speech. Measured in isolation, the first chunk was ready **0.60 s** after the prompt.

### Caveat on the token-rate figures

The `4.0 tok/s` minimum is an artifact, not a regression. Very short replies (3 tokens) are
dominated by prompt processing, so dividing by token count understates the rate. The **median
14.4 tok/s** on 10–22 token replies is the honest number, and matches the 14.95 tok/s measured
in the standalone LLM benchmark.

---

## 2. Why the NPU is required for this

The same Whisper-small model on CPU measured **RTF 1.21** (see `docs/asr/01`) — slower than
real-time. A conversational agent is not possible on the CPU path on this part: transcription
alone would fall behind the speaker before the LLM was even invoked.

On the NPU, ASR consumes **RTF 0.12** of the budget, leaving the rest for generation and speech
synthesis.

---

## 3. Issues encountered

### Issue 1 — ONNX Runtime and onnxruntime-genai cannot both load models in one process

**The most significant constraint found.** Loading a VitisAI EP model (Whisper) and an OGA
model (Phi) in the same process aborts:

```
[libprotobuf ERROR google/protobuf/descriptor_database.cc:120]
    File already exists in database: external_data.proto
[libprotobuf FATAL] CHECK failed: GeneratedDatabase()->Add(...)
```

Both libraries statically link protobuf and each registers `external_data.proto`; the second
registration is fatal and the process wedges rather than raising a Python exception.

**Misleading detail:** *importing* both modules works fine —
`import onnxruntime` then `import onnxruntime_genai` succeeds and reports
`['VitisAIExecutionProvider', 'CPUExecutionProvider']`. The collision only occurs at **model
load**. An import-level feasibility check gives a false pass.

**Workaround.** Run the LLM in a **separate process** and talk to it over stdin/stdout JSON
(`scripts/voice-agent/llm_server.py`). This also keeps the model resident between turns, so
there is no reload cost per question.

**Implication beyond this demo:** any application combining a VitisAI EP workload (CNN, ASR)
with an OGA LLM on this stack must use process isolation.

### Issue 2 — AMD's Whisper demo has a broken microphone path

`run_whisper.py --input mic` fails immediately:

```
NameError: name 'sd' is not defined
```

`import sounddevice as sd` sits at **line 357, inside `main()`**, making it function-local.
`mic_stream()` and its inner `feeder()` are module-level and reference a global `sd` that never
exists. Mic mode cannot work as shipped.

**Fix:** move the import to module scope. Our patch also makes the silence parameters
overridable (see Issue 3). The unmodified original is kept at
the original AMD `run_whisper.py`. It is not redistributed in this repository;
obtain it with the AMD Ryzen AI ASR example.

### Issue 3 — 5-second silence timeout is too aggressive for live use

`mic_stream()` hard-codes `silence_duration=5.0` and exits on the first 5 s of quiet. Since
model load takes ~7 s, the demo frequently ended before the speaker could react.

**Fix:** parameterised via `ASR_SILENCE_THRESHOLD` / `ASR_SILENCE_DURATION`. Our own recorder
(`voice_assistant.py`) additionally does **not arm the silence cutoff until speech has actually
been detected**, so it waits indefinitely for you to start.

### Issue 4 — PortAudio missing

`sounddevice` is in `requirements.txt` but its native dependency is not:

```
OSError: PortAudio library not found
```

**Fix:** `sudo apt install libportaudio2`. Not mentioned in AMD's README.

### Issue 5 — the default input device is not the USB headset

With the Jabra connected, the default PipeWire source remained the **internal microphone**.
The demo opens the *default* device and never accepts a device argument, so it silently
recorded from the wrong microphone.

Confirmed by RMS comparison: default source 0.03637 vs Jabra 0.00476 in the same quiet room.

**Fix:**

```bash
wpctl status                      # find the Jabra source ID
wpctl set-default <id>
```

Verify by re-reading RMS from the default device and confirming it matches the Jabra directly.

### Issue 6 — espeak-ng sounds robotic

`spd-say` (speech-dispatcher → espeak-ng) is preinstalled and works, but is obviously
synthetic and undermines a demo.

**Fix:** Piper neural TTS in an **isolated venv** (`/home/amd/tts-venv`), voice
`en_US-amy-medium` (63 MB). Audio is streamed raw from Piper into `aplay` rather than through a
temp file, so playback begins during synthesis. `spd-say` remains an automatic fallback.

Piper is deliberately isolated for the same reason as Quark: it pulls its own `onnxruntime`,
which must not disturb the Ryzen AI stack (see `docs/03`, Problem 7).

### Issue 7 — the model confabulates, confidently

Not a defect, but it will mislead an audience:

| Question | Reply | Reality |
|---|---|---|
| "What is the current temperature in New York City?" | "…is 58°F (14°C)." | **Invented.** No network, no weather data. |
| "Can you change the robotic voice to something else?" | "Switching to a more human-like voice tone now." | **Invented.** It has no control over TTS. |

Use factual, static questions for demos (capitals, arithmetic, definitions). Avoid anything
time-sensitive or self-referential.

### Issue 8 — ASR errors propagate silently

"What is the **capital** of Canada?" was transcribed as "What is the **cap** of Canada?", and
the LLM answered about the **Canadian dollar**. The pipeline has no confidence signal — a
mistranscription becomes a confident wrong answer with nothing to flag it.

### Issue 9 — empty transcriptions

7 of 24 turns produced empty text after detecting speech. Likely the tail of the agent's own
playback re-triggering the recorder, or brief noise crossing the RMS threshold. Harmless (the
agent says "I did not catch that") but it shortens usable turns. Not root-caused; a proper fix
would gate the recorder while playback is active.

---

## 4. Running it

```bash
# one-off prerequisites
sudo apt install libportaudio2
python3.12 -m venv /home/amd/tts-venv && /home/amd/tts-venv/bin/pip install piper-tts
mkdir -p /home/amd/piper-voices && cd /home/amd/piper-voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

# make the Jabra the default input and output
wpctl status && wpctl set-default <jabra-source-id>

# run (from the directory holding AMD's run_whisper.py and config/)
source scripts/asr/voice-assistant-env.sh
python -u voice_assistant.py \
  --model-type whisper-small --device npu \
  --llm-model /home/amd/run_llm/Phi-3.5-mini-instruct_rai_1.8.0_npu_4K \
  --llm-env scripts/asr/voice-assistant-env.sh \
  --turns 0 --silence-seconds 1.0 --startup-grace 300 --max-new-tokens 40
```

`voice-assistant-env.sh` is required rather than `ryzenai-env.sh`: it adds **both** the peano
path (VitisAI EP) and `deployment/lib` + `RYZENAI_EP_PATH` (OGA), which no single AMD-documented
environment provides.

`--turns 0` runs until Ctrl-C.

---

## 5. Artefacts

| Item | Path |
|---|---|
| Voice agent (ASR + LLM + TTS) | `scripts/voice-agent/voice_assistant.py` |
| Isolated LLM worker | `scripts/voice-agent/llm_server.py` |
| Transcribe-and-echo demo (no LLM) | `scripts/voice-agent/voice_loop.py` |
| Unmodified AMD demo, for diffing | Obtain `run_whisper.py` from AMD's Ryzen AI ASR example |
| Combined environment | `scripts/asr/voice-assistant-env.sh` |

---

## 6. Limitations

- **Latency figures are from one session** of 17 turns, single speaker, quiet room. No repeat
  runs or speaker variation.
- **No power measurement for the combined pipeline.** ASR and LLM were measured separately
  (`docs/asr/01`, `docs/llm/01`); the agent as a whole has not been instrumented.
- **No accuracy measurement.** No WER for the ASR stage, no evaluation of answer quality.
- **Single-turn context.** Each question is independent; there is no conversation history.
- **No barge-in.** The agent cannot be interrupted while speaking.
