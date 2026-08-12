"""LLM worker process for the voice assistant.

Runs Phi-3.5-mini on the NPU via onnxruntime-genai and answers JSON-lines
requests on stdin.

This exists as a SEPARATE PROCESS because onnxruntime and onnxruntime-genai each
statically link protobuf and both register external_data.proto. Loading models
from both in one process aborts with:

    [libprotobuf FATAL] CHECK failed: GeneratedDatabase()->Add(...)
    File already exists in database: external_data.proto

Importing both is fine; it is the second model load that dies. Process isolation
is therefore required, not merely tidier.

Protocol: one JSON object per line in, one per line out.
    in : {"prompt": "..."}            out: {"reply": "...", "tokens": n, "seconds": s}
    in : {"quit": true}               out: (exits)
"""

from __future__ import annotations

import argparse
import json
import sys
import time

SYSTEM = (
    "You are a voice assistant. Reply with ONE short spoken sentence, under 20 words. "
    "No lists, no markdown, no preamble."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    args = parser.parse_args()

    import onnxruntime_genai as og

    model = og.Model(args.model)
    tokenizer = og.Tokenizer(model)
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if request.get("quit"):
            break

        prompt = request.get("prompt", "").strip()
        if not prompt:
            print(json.dumps({"reply": "", "tokens": 0, "seconds": 0.0}), flush=True)
            continue

        chat = f"<|system|>\n{SYSTEM}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        tokens = tokenizer.encode(chat)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(tokens) + args.max_new_tokens, do_sample=False)
        generator = og.Generator(model, params)
        generator.append_tokens(tokens)

        started = time.perf_counter()
        produced: list[int] = []
        spoken_upto = 0
        first_chunk_at = None

        def clean(text: str) -> str:
            for marker in ("<|end|>", "<|endoftext|>", "<|user|>", "<|system|>"):
                text = text.split(marker)[0]
            return text

        # Emit each finished sentence as soon as it exists so the caller can start
        # speaking while the rest is still being generated.
        while not generator.is_done() and len(produced) < args.max_new_tokens:
            generator.generate_next_token()
            produced.append(generator.get_next_tokens()[0])
            decoded = clean(tokenizer.decode(produced))
            boundary = max(decoded.rfind(c) for c in ".!?")
            if boundary > spoken_upto:
                chunk = decoded[spoken_upto : boundary + 1].strip()
                if len(chunk) >= 2:
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter() - started
                    print(json.dumps({"chunk": chunk}), flush=True)
                    spoken_upto = boundary + 1

        elapsed = time.perf_counter() - started
        reply = clean(tokenizer.decode(produced)).strip()
        tail = reply[spoken_upto:].strip()
        if tail:
            if first_chunk_at is None:
                first_chunk_at = elapsed
            print(json.dumps({"chunk": tail}), flush=True)
        print(
            json.dumps(
                {
                    "reply": reply,
                    "tokens": len(produced),
                    "seconds": elapsed,
                    "first_chunk_seconds": first_chunk_at,
                }
            ),
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
