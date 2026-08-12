"""Voice loop demo: speak into the Jabra, hear the NPU transcription read back.

Reuses AMD's WhisperONNX class from run_whisper.py (so inference is the same NPU
path as the demo) and adds record-until-silence plus spoken playback.

Playback uses spd-say (speech-dispatcher / espeak-ng), which is already present
on Ubuntu 24.04 and follows the default PipeWire sink -- so it comes out of the
Jabra when the Jabra is the default output.

Run from the run_asr directory:
    python voice_loop.py --model-type whisper-small --device npu
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent))

SAMPLE_RATE = 16000


def speak(text: str, rate: int = 0) -> None:
    """Blocking TTS via speech-dispatcher; -w waits until playback finishes."""
    subprocess.run(["spd-say", "-w", "-r", str(rate), text], check=False)


def record_until_silence(
    silence_threshold: float,
    silence_seconds: float,
    max_seconds: float,
    startup_grace: float,
) -> np.ndarray:
    """Capture from the default input until the speaker stops talking."""
    frames: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, _frames, _time, status):
        if status:
            print(f"  (audio status: {status})", file=sys.stderr)
        frames.put(indata.copy())

    collected: list[np.ndarray] = []
    started = time.time()
    last_voice = None
    heard_anything = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        while True:
            try:
                block = frames.get(timeout=0.5)
            except queue.Empty:
                block = None
            now = time.time()
            if block is not None:
                collected.append(block)
                rms = float(np.sqrt(np.mean(block**2)))
                if rms >= silence_threshold:
                    if not heard_anything:
                        print("  ...speech detected")
                    heard_anything = True
                    last_voice = now
            if now - started > max_seconds:
                print("  ...max duration reached")
                break
            # Do not arm the silence cutoff until the speaker has actually begun.
            if heard_anything and last_voice is not None and now - last_voice >= silence_seconds:
                print("  ...silence, stopping")
                break
            if not heard_anything and now - started > startup_grace:
                print("  ...no speech heard")
                break

    if not collected:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(collected, axis=0).reshape(-1).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="whisper-small")
    parser.add_argument("--device", default="npu", choices=("npu", "cpu"))
    parser.add_argument("--config-file", default="./config/model_config.json")
    parser.add_argument("--silence-threshold", type=float, default=0.01)
    parser.add_argument("--silence-seconds", type=float, default=2.5)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--startup-grace", type=float, default=20.0)
    parser.add_argument("--turns", type=int, default=0, help="0 = loop until Ctrl-C")
    parser.add_argument("--prefix", default="You said:")
    args = parser.parse_args()

    from run_whisper import WhisperONNX, download_whisper_onnx, load_provider_options

    with open(args.config_file) as handle:
        config = json.load(handle)
    encoder_options, decoder_options = load_provider_options(config, args.model_type, args.device)
    encoder_path, decoder_path = download_whisper_onnx(args.model_type)

    print(f"Loading {args.model_type} on {args.device} (first run compiles, please wait) ...")
    load_started = time.perf_counter()
    model = WhisperONNX(
        encoder_path,
        decoder_path,
        args.model_type,
        encoder_providers=encoder_options,
        decoder_providers=decoder_options,
    )
    print(f"Model ready in {time.perf_counter() - load_started:.1f} s")

    speak("Ready. Say something after the beep.")
    print("\n=== voice loop: Ctrl-C to quit ===")

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
                print("  nothing captured; still listening")
                continue

            started = time.perf_counter()
            text, _ = model.transcribe(audio, is_mic=True)
            elapsed = time.perf_counter() - started
            text = (text or "").strip()
            rtf = elapsed / seconds if seconds else 0.0
            print(f"  audio {seconds:.2f}s | inference {elapsed:.2f}s | RTF {rtf:.2f}")
            print(f"  >>> {text!r}")

            if text:
                speak(f"{args.prefix} {text}")
            else:
                speak("I did not catch that.")
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
