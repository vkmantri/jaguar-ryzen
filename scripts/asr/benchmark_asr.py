"""Benchmark AMD's Whisper ASR demo with power and NPU verification.

Separate from the CNN and LLM harnesses: run_whisper.py is AMD's own script, so
this wraps it as a subprocess and parses its output.

run_whisper.py does not raise ONNX Runtime's log severity, so no VitisAI EP
partitioning markers are emitted. NPU residency is therefore verified from the
driver's hardware contexts, which works regardless of log level or cache warmth.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

XRT_SMI = "/opt/xilinx/xrt/bin/xrt-smi"

_TTFT = re.compile(r"Time to First Token[^:]*:\s*([\d.]+)\s*seconds")
_RTF = re.compile(r"RTF:\s*([\d.]+)")
_TRANSCRIPTION = re.compile(r"^Transcription:\s*(.+)$", re.MULTILINE)
_CHUNK = re.compile(r"Performance Metric \(Chunk (\d+)\)")


def read_power_microwatts() -> dict[str, int]:
    readings: dict[str, int] = {}
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            name = (hwmon / "name").read_text().strip()
        except OSError:
            continue
        for rail in sorted(hwmon.glob("power*_input")):
            try:
                readings[f"{name}.{rail.name}"] = int(rail.read_text().strip())
            except (OSError, ValueError):
                continue
    return readings


def read_active_contexts() -> list[dict[str, object]]:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "aie.json"
        try:
            subprocess.run(
                [XRT_SMI, "examine", "--report", "aie-partitions", "-f", "JSON", "-o", str(report), "--force"],
                capture_output=True,
                timeout=30,
                check=False,
            )
            document = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError, subprocess.SubprocessError):
            return []
    contexts = []
    for device in document.get("devices", []):
        for partition in device.get("aie_partitions", {}).get("partitions", []):
            for context in partition.get("hw_contexts", []):
                if str(context.get("status")) == "Active":
                    contexts.append(
                        {
                            "pid": context.get("pid"),
                            "command_submissions": context.get("command_submissions"),
                            "command_completions": context.get("command_completions"),
                            "gops": context.get("gops"),
                            "errors": context.get("errors"),
                        }
                    )
    return contexts


def parse_output(text: str) -> dict[str, object]:
    ttft = [float(v) for v in _TTFT.findall(text)]
    rtf = [float(v) for v in _RTF.findall(text)]
    transcriptions = [t.strip() for t in _TRANSCRIPTION.findall(text)]
    result: dict[str, object] = {
        "chunks": len(_CHUNK.findall(text)),
        "time_to_first_token_seconds": ttft,
        "real_time_factor": rtf,
        "transcription": transcriptions,
    }
    if ttft:
        result["ttft_mean_seconds"] = statistics.fmean(ttft)
    if rtf:
        result["rtf_mean"] = statistics.fmean(rtf)
        result["speedup_vs_realtime"] = 1.0 / statistics.fmean(rtf) if statistics.fmean(rtf) else None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="whisper-small")
    parser.add_argument("--device", default="npu", choices=("npu", "cpu"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--label", default="asr_whisper")
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--script", default="run_whisper.py")
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = args.results_root / f"{timestamp}_{args.label}_{secrets.token_hex(4)}"
    run.mkdir(parents=True, exist_ok=False)

    idle_power = read_power_microwatts()
    samples: list[dict[str, object]] = []
    contexts: list[dict[str, object]] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            samples.append({"at": time.time(), "power_uw": read_power_microwatts()})
            contexts.extend(read_active_contexts())
            stop.wait(args.sample_interval)

    command = [
        "python", args.script,
        "--model-type", args.model_type,
        "--device", args.device,
        "--input", args.input,
    ]
    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    wall_seconds = time.perf_counter() - started
    stop.set()
    worker.join(timeout=10)

    output = completed.stdout + completed.stderr
    (run / "run_whisper.log").write_text(output, encoding="utf-8")

    def summarise(rail: str) -> dict[str, float] | None:
        values = [
            s["power_uw"][rail] / 1e6  # type: ignore[index]
            for s in samples
            if rail in s["power_uw"]  # type: ignore[operator]
        ]
        if not values:
            return None
        return {"mean_w": statistics.fmean(values), "max_w": max(values), "samples": len(values)}

    rails = sorted({k for s in samples for k in s["power_uw"]})  # type: ignore[union-attr]
    document = {
        "run_id": run.name,
        "kind": "asr_whisper_benchmark",
        "command": command,
        "exit_code": completed.returncode,
        "wall_seconds": wall_seconds,
        "model_type": args.model_type,
        "device": args.device,
        "audio_input": args.input,
        "metrics": parse_output(output),
        "power": {rail: summarise(rail) for rail in rails},
        "idle_power_uw": idle_power,
        "npu_verification": {
            "executed_on_npu": bool(contexts),
            "reason": (
                f"driver reported {len(contexts)} active NPU hardware context sample(s)"
                if contexts
                else "no active NPU hardware context observed"
            ),
            "max_command_submissions": max(
                (int(c["command_submissions"]) for c in contexts if c.get("command_submissions") is not None),
                default=0,
            ),
            "sample_count": len(samples),
        },
        "completed_at": datetime.now(UTC).isoformat(),
    }
    (run / "asr_benchmark.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(run)
    return 0 if completed.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
