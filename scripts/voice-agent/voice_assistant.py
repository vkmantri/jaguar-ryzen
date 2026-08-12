"""Voice assistant: speak to the Jabra, get a spoken answer. Both models on the NPU.

    Jabra mic -> Whisper (NPU, VitisAI EP) -> text
              -> Phi-3.5-mini (NPU, OGA RyzenAI EP) -> reply
              -> spd-say -> Jabra speaker

Whisper alone is speech-to-text and cannot answer anything; the LLM stage is what
makes this a conversation rather than an echo.

Both runtimes load in one process, but the environment must satisfy both: the LLM
needs deployment/lib and RYZENAI_EP_PATH, while the VitisAI EP additionally needs
the peano library path. See scripts/asr/voice-assistant-env.sh.

Run from the run_asr directory (it imports AMD's run_whisper.py):
    python voice_assistant.py --llm-model /home/amd/run_llm/Phi-3.5-mini-instruct_rai_1.8.0_npu_4K
"""

from __future__ import annotations

import argparse
import json
import queue
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent))

SAMPLE_RATE = 16000


PIPER_BIN = "/home/amd/tts-venv/bin/piper"
PIPER_VOICE = "/home/amd/piper-voices/en_US-amy-medium.onnx"
PIPER_RATE = 22050


def speak(text: str) -> None:
    """Piper neural TTS streamed straight to aplay; espeak via spd-say as fallback."""
    if Path(PIPER_BIN).exists() and Path(PIPER_VOICE).exists():
        piper = subprocess.Popen(
            [PIPER_BIN, "-m", PIPER_VOICE, "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        player = subprocess.Popen(
            ["aplay", "-q", "-r", str(PIPER_RATE), "-f", "S16_LE", "-t", "raw", "-c", "1"],
            stdin=piper.stdout,
            stderr=subprocess.DEVNULL,
        )
        piper.stdout.close()  # type: ignore[union-attr]
        piper.communicate(text.encode())
        player.wait()
        return
    subprocess.run(["spd-say", "-w", text], check=False)


def record_until_silence(
    silence_threshold: float, silence_seconds: float, max_seconds: float, startup_grace: float
) -> np.ndarray:
    frames: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, _frames, _time, status):
        frames.put(indata.copy())

    collected: list[np.ndarray] = []
    started = time.time()
    last_voice = None
    heard = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        while True:
            try:
                block = frames.get(timeout=0.5)
            except queue.Empty:
                block = None
            now = time.time()
            if block is not None:
                collected.append(block)
                if float(np.sqrt(np.mean(block**2))) >= silence_threshold:
                    if not heard:
                        print("  ...speech detected")
                    heard = True
                    last_voice = now
            if now - started > max_seconds:
                break
            if heard and last_voice is not None and now - last_voice >= silence_seconds:
                print("  ...silence, stopping")
                break
            if not heard and now - started > startup_grace:
                print("  ...no speech heard")
                break

    if not collected:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(collected, axis=0).reshape(-1).astype(np.float32)


class Responder:
    """Phi-3.5-mini on the NPU, in a separate process.

    onnxruntime and onnxruntime-genai both statically link protobuf and register
    external_data.proto, so loading models from both in one process aborts with
    a libprotobuf FATAL. Isolation is required, not just tidier.
    """

    def __init__(self, model_path: str, max_new_tokens: int, worker: str, env_script: str) -> None:
        command = (
            f"source {shlex.quote(env_script)} >/dev/null 2>&1; "
            f"exec python -u {shlex.quote(worker)} --model {shlex.quote(model_path)} "
            f"--max-new-tokens {max_new_tokens}"
        )
        self.process = subprocess.Popen(
            ["bash", "-lc", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        while True:
            line = self.process.stdout.readline()  # type: ignore[union-attr]
            if not line:
                raise RuntimeError("LLM worker exited before becoming ready")
            try:
                if json.loads(line).get("ready"):
                    break
            except json.JSONDecodeError:
                continue

    def reply(self, prompt: str, on_chunk=None) -> tuple[str, float, int, float | None]:
        self.process.stdin.write(json.dumps({"prompt": prompt}) + "\n")  # type: ignore[union-attr]
        self.process.stdin.flush()  # type: ignore[union-attr]
        while True:
            line = self.process.stdout.readline()  # type: ignore[union-attr]
            if not line:
                raise RuntimeError("LLM worker exited unexpectedly")
            try:
                answer = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Sentence chunks arrive before the final record so playback can start early.
            if "chunk" in answer:
                if on_chunk is not None:
                    on_chunk(answer["chunk"])
                continue
            if "reply" in answer:
                return (
                    answer["reply"],
                    float(answer["seconds"]),
                    int(answer["tokens"]),
                    answer.get("first_chunk_seconds"),
                )

    def close(self) -> None:
        try:
            self.process.stdin.write(json.dumps({"quit": True}) + "\n")  # type: ignore[union-attr]
            self.process.stdin.flush()  # type: ignore[union-attr]
            self.process.wait(timeout=10)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            if self.process.poll() is None:
                self.process.kill()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="whisper-small")
    parser.add_argument("--device", default="npu", choices=("npu", "cpu"))
    parser.add_argument("--config-file", default="./config/model_config.json")
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-worker", default="llm_server.py")
    parser.add_argument(
        "--llm-env",
        default="/home/amd/workspace/jaguar-eval/scripts/asr/voice-assistant-env.sh",
        help="environment script sourced by the isolated LLM worker (must not require a specific cwd)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--silence-threshold", type=float, default=0.01)
    parser.add_argument("--silence-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--startup-grace", type=float, default=25.0)
    parser.add_argument("--turns", type=int, default=0)
    args = parser.parse_args()

    from run_whisper import WhisperONNX, download_whisper_onnx, load_provider_options

    with open(args.config_file) as handle:
        config = json.load(handle)
    encoder_options, decoder_options = load_provider_options(config, args.model_type, args.device)
    encoder_path, decoder_path = download_whisper_onnx(args.model_type)

    print(f"Loading {args.model_type} (ASR) on {args.device} ...")
    started = time.perf_counter()
    asr = WhisperONNX(
        encoder_path, decoder_path, args.model_type,
        encoder_providers=encoder_options, decoder_providers=decoder_options,
    )
    print(f"  ASR ready in {time.perf_counter() - started:.1f} s")

    print("Loading Phi-3.5-mini (LLM) on NPU in an isolated process ...")
    started = time.perf_counter()
    llm = Responder(args.llm_model, args.max_new_tokens, args.llm_worker, args.llm_env)
    print(f"  LLM ready in {time.perf_counter() - started:.1f} s")

    speak("Assistant ready. Ask me something.")
    print("\n=== voice assistant: Ctrl-C to quit ===")

    turn = 0
    try:
        while args.turns == 0 or turn < args.turns:
            turn += 1
            print(f"\n[turn {turn}] listening ...")
            audio = record_until_silence(
                args.silence_threshold, args.silence_seconds, args.max_seconds, args.startup_grace
            )
            seconds = len(audio) / SAMPLE_RATE
            if seconds < 0.3:
                continue

            started = time.perf_counter()
            heard, _ = asr.transcribe(audio, is_mic=True)
            asr_seconds = time.perf_counter() - started
            heard = (heard or "").strip()
            print(f"  heard ({asr_seconds:.2f}s, RTF {asr_seconds / seconds:.2f}): {heard!r}")
            if not heard:
                speak("I did not catch that.")
                continue

            # Speaking each sentence as it arrives hides most of the generation time.
            spoke_any = False
            speech_started = None

            def say_chunk(chunk: str) -> None:
                nonlocal spoke_any, speech_started
                if speech_started is None:
                    speech_started = time.perf_counter()
                spoke_any = True
                print(f"  ...speaking: {chunk!r}")
                speak(chunk)

            turn_started = time.perf_counter()
            answer, llm_seconds, tokens, _first_chunk = llm.reply(heard, on_chunk=say_chunk)
            rate = tokens / llm_seconds if llm_seconds else 0.0
            latency = (speech_started - turn_started) if speech_started else llm_seconds
            print(
                f"  reply ({llm_seconds:.2f}s, {tokens} tok, {rate:.1f} tok/s | "
                f"first audio at {latency:.2f}s): {answer!r}"
            )
            if not spoke_any:
                speak(answer or "Sorry, I have no answer.")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
