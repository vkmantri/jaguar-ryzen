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
PIPER_OUT = "/tmp/voice_agent_tts"


class Speaker:
    """Resident Piper writing one WAV per utterance, played back synchronously.

    Two constraints have to hold at once. Respawning piper per sentence reloads
    the 63 MB voice model (0.99 s vs 0.13 s warm), so the process must persist.
    But playback must also block: if it does not, the recorder reopens while the
    reply is still audible, captures it, and every following turn transcribes to
    nothing. Writing per-utterance WAVs gives both a warm model and a clean
    boundary to wait on.
    """

    def __init__(self) -> None:
        self.piper = None
        self.out = Path(PIPER_OUT)
        if not (Path(PIPER_BIN).exists() and Path(PIPER_VOICE).exists()):
            return
        self.out.mkdir(parents=True, exist_ok=True)
        for stale in self.out.glob("*.wav"):
            stale.unlink()
        self.piper = subprocess.Popen(
            [PIPER_BIN, "-m", PIPER_VOICE, "-d", str(self.out), "--output-dir-naming", "timestamp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def warm_up(self) -> None:
        if self.piper is not None:
            self.say("Ready.", play=False)

    def say(self, text: str, play: bool = True) -> None:
        text = text.strip()
        if not text:
            return
        if self.piper is None:
            subprocess.run(["spd-say", "-w", text], check=False)
            return
        before = {p.name for p in self.out.glob("*.wav")}
        self.piper.stdin.write(text + "\n")  # type: ignore[union-attr]
        self.piper.stdin.flush()  # type: ignore[union-attr]

        wav = self._wait_for_wav(before)
        if wav is None:
            return
        if play:
            subprocess.run(["aplay", "-q", str(wav)], stderr=subprocess.DEVNULL, check=False)
        wav.unlink(missing_ok=True)

    def _wait_for_wav(self, before: set[str], timeout: float = 20.0) -> Path | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            new = [p for p in self.out.glob("*.wav") if p.name not in before]
            if new:
                wav = new[0]
                last = -1
                while time.time() < deadline:
                    size = wav.stat().st_size
                    if size == last and size > 44:
                        return wav
                    last = size
                    time.sleep(0.01)
                return wav
            time.sleep(0.005)
        return None

    def close(self) -> None:
        if self.piper is None:
            return
        try:
            if self.piper.stdin:
                self.piper.stdin.close()
            self.piper.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.piper.kill()


def record_until_silence(
    silence_threshold: float, silence_seconds: float, max_seconds: float, startup_grace: float
) -> tuple[np.ndarray, float]:
    """Returns the audio and the perf_counter timestamp at which speech stopped."""
    frames: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, _frames, _time, status):
        frames.put(indata.copy())

    collected: list[np.ndarray] = []
    started = time.time()
    last_voice = None
    last_voice_perf = None
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
                    last_voice_perf = time.perf_counter()
            if now - started > max_seconds:
                break
            if heard and last_voice is not None and now - last_voice >= silence_seconds:
                print("  ...silence, stopping")
                break
            if not heard and now - started > startup_grace:
                print("  ...no speech heard")
                break

    if not collected:
        return np.zeros(0, dtype=np.float32), time.perf_counter()
    ended = last_voice_perf if last_voice_perf is not None else time.perf_counter()
    return np.concatenate(collected, axis=0).reshape(-1).astype(np.float32), ended


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
    parser.add_argument("--silence-seconds", type=float, default=0.8)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--startup-grace", type=float, default=25.0)
    parser.add_argument("--turns", type=int, default=0)
    parser.add_argument(
        "--wake-word",
        # Whisper transcribes the brand phonetically, so accept the homophones too.
        default="hey deere,hey dear,hey deer,hello deere,hello dear,hello deer,ok deere,ok dear",
        help="comma-separated wake phrases, matched case-insensitively",
    )
    parser.add_argument(
        "--sleep-after",
        type=float,
        default=60.0,
        help="seconds of no successful interaction before requiring the wake word",
    )
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

    speaker = Speaker()
    started = time.perf_counter()
    speaker.warm_up()
    print(f"  TTS ready in {time.perf_counter() - started:.1f} s")

    speaker.say("Deere ready. Ask me something.")
    print("\n=== voice assistant: Ctrl-C to quit ===")

    wake_words = [w.strip().lower() for w in args.wake_word.split(",") if w.strip()]

    def is_wake(text: str) -> bool:
        low = text.lower()
        return any(w in low for w in wake_words)

    def strip_wake(text: str) -> str:
        low = text.lower()
        for w in wake_words:
            index = low.find(w)
            if index != -1:
                return (text[:index] + text[index + len(w) :]).strip(" ,.!?")
        return text

    awake = True
    last_interaction = time.monotonic()
    turn = 0
    try:
        while args.turns == 0 or turn < args.turns:
            turn += 1
            if awake and time.monotonic() - last_interaction > args.sleep_after:
                awake = False
                print(f"  (idle {args.sleep_after:.0f}s -- sleeping, say '{wake_words[0]}' to wake)")
            state = "listening" if awake else "asleep"
            print(f"\n[turn {turn}] {state} ...")
            audio, speech_ended = record_until_silence(
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

            if not awake:
                if not is_wake(heard):
                    continue  # stay silent; ambient noise must not trigger a reply
                awake = True
                last_interaction = time.monotonic()
                heard = strip_wake(heard)
                if not heard:
                    speaker.say("Yes?")
                    continue
            elif is_wake(heard):
                heard = strip_wake(heard) or heard

            if not heard:
                # Silence and noise are common; only answer when there were words.
                continue
            last_interaction = time.monotonic()

            # Speaking each sentence as it arrives hides most of the generation time.
            spoke_any = False
            speech_started = None

            def say_chunk(chunk: str) -> None:
                nonlocal spoke_any, speech_started
                if speech_started is None:
                    speech_started = time.perf_counter()
                spoke_any = True
                print(f"  ...speaking: {chunk!r}")
                speaker.say(chunk)

            answer, llm_seconds, tokens, _first_chunk = llm.reply(heard, on_chunk=say_chunk)
            rate = tokens / llm_seconds if llm_seconds else 0.0
            if not spoke_any:
                speech_started = time.perf_counter()
                speaker.say(answer or "Sorry, I have no answer.")
            # Measured from when the speaker stopped talking, so it matches a stopwatch.
            end_to_end = (speech_started or time.perf_counter()) - speech_ended
            print(
                f"  reply ({llm_seconds:.2f}s, {tokens} tok, {rate:.1f} tok/s): {answer!r}"
            )
            print(
                f"  END-TO-END speech-end -> audio-out: {end_to_end:.2f}s "
                f"(silence {args.silence_seconds:.1f} + asr {asr_seconds:.2f} + llm {llm_seconds:.2f})"
            )
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        llm.close()
        speaker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
