"""Benchmark the Ryzen AI LLM (OGA) flow with power and NPU verification.

Kept separate from the CNN harness: model_benchmark is a prebuilt C++ binary, so
this wraps it as a subprocess rather than driving onnxruntime directly.

model_benchmark has no NPU execution-provider flag -- the provider is selected by
RYZENAI_EP_PATH and the model's genai_config.json -- so NPU residency is verified
out-of-band from the driver's hardware contexts, exactly as for the CNN runs.
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

_METRIC_BLOCK = re.compile(
    r"^(?P<name>[A-Z][^:\n]*):\s*\n(?P<body>(?:\s+\S+.*\n)+)", re.MULTILINE
)
_METRIC_LINE = re.compile(r"^\s+(?P<key>[a-z0-9_ ()/]+):\s+(?P<value>[\d.eE+-]+)\s*$", re.MULTILINE)
_PEAK = re.compile(r"Peak working set size \(bytes\):\s*(\d+)")
_PROMPT_TOKENS = re.compile(r"Prompt Number of Tokens:\s*(\d+)")


def read_power_microwatts() -> dict[str, int]:
    """Read hwmon power rails by sensor name; hwmon indices are not stable."""
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


def read_hardware_contexts() -> list[dict[str, object]]:
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
                contexts.append(
                    {
                        "pid": context.get("pid"),
                        "status": context.get("status"),
                        "command_submissions": context.get("command_submissions"),
                        "command_completions": context.get("command_completions"),
                        "gops": context.get("gops"),
                        "errors": context.get("errors"),
                        "num_cols": partition.get("num_cols"),
                    }
                )
    return contexts


def parse_benchmark_output(text: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for block in _METRIC_BLOCK.finditer(text):
        name = block.group("name").strip()
        values = {
            line.group("key").strip(): float(line.group("value"))
            for line in _METRIC_LINE.finditer(block.group("body"))
        }
        if values:
            metrics[name] = values
    peak = _PEAK.search(text)
    if peak:
        metrics["peak_working_set_bytes"] = int(peak.group(1))
    tokens = _PROMPT_TOKENS.search(text)
    if tokens:
        metrics["prompt_number_of_tokens"] = int(tokens.group(1))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--generation-length", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--label", default="llm_phi35")
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--binary", default="./model_benchmark")
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
            found = read_hardware_contexts()
            if found:
                contexts.extend(found)
            stop.wait(args.sample_interval)

    command = [
        args.binary,
        "-i", args.model,
        "-l", str(args.prompt_length),
        "-g", str(args.generation_length),
        "-r", str(args.repetitions),
    ]
    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    wall_seconds = time.perf_counter() - started
    stop.set()
    worker.join(timeout=10)

    output = completed.stdout + completed.stderr
    (run / "model_benchmark.log").write_text(output, encoding="utf-8")

    def summarise(key: str) -> dict[str, float] | None:
        values = [s["power_uw"].get(key) for s in samples if key in s["power_uw"]]  # type: ignore[union-attr]
        values = [v / 1e6 for v in values if v is not None]
        if not values:
            return None
        return {
            "mean_w": statistics.fmean(values),
            "max_w": max(values),
            "min_w": min(values),
            "samples": len(values),
        }

    rails = sorted({k for s in samples for k in s["power_uw"]})  # type: ignore[union-attr]
    active_contexts = [c for c in contexts if str(c.get("status")) == "Active"]
    document = {
        "run_id": run.name,
        "kind": "llm_oga_benchmark",
        "command": command,
        "exit_code": completed.returncode,
        "wall_seconds": wall_seconds,
        "model": args.model,
        "metrics": parse_benchmark_output(output),
        "power": {rail: summarise(rail) for rail in rails},
        "idle_power_uw": idle_power,
        "npu_verification": {
            "executed_on_npu": bool(active_contexts),
            "reason": (
                f"driver reported {len(active_contexts)} active NPU hardware context sample(s)"
                if active_contexts
                else "no active NPU hardware context observed; treat as CPU execution"
            ),
            "max_command_submissions": max(
                (int(c["command_submissions"]) for c in active_contexts if c.get("command_submissions") is not None),
                default=0,
            ),
            "max_gops": max(
                (int(c["gops"]) for c in active_contexts if str(c.get("gops")).isdigit()),
                default=0,
            ),
            "sample_count": len(samples),
        },
        "completed_at": datetime.now(UTC).isoformat(),
    }
    (run / "llm_benchmark.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(run)
    return 0 if completed.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
