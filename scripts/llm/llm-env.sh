#!/usr/bin/env bash
# Environment for the Ryzen AI LLM (OGA) flow. Source it, do not execute:
#   cd /home/amd/run_llm && source /path/to/llm-env.sh
#
# Separate from scripts/ryzenai-env.sh on purpose: the LLM flow uses the
# standalone `deployment/` tree and the RyzenAI GenAI provider, not the VitisAI
# ONNX Runtime EP used for the CNN work.
#
# Must be sourced from the run_llm working directory (the one holding
# deployment/, model_benchmark and the model folder).

RYZENAI_VENV="${RYZENAI_VENV:-/home/amd/ryzenai/venv}"

if [[ ! -f "$RYZENAI_VENV/bin/activate" ]]; then
    echo "ERROR: no venv at $RYZENAI_VENV (set RYZENAI_VENV)" >&2
    return 1
fi
if [[ ! -d "deployment/lib" ]]; then
    echo "ERROR: run this from the directory containing deployment/ (currently $PWD)" >&2
    return 1
fi

# Runtime-managed files intentionally live outside this repository.
# shellcheck disable=SC1091
source "$RYZENAI_VENV/bin/activate"
# shellcheck disable=SC1091
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$PWD/deployment/lib:$LD_LIBRARY_PATH
export RYZENAI_EP_PATH=$PWD/deployment/lib/libonnxruntime_providers_ryzenai.so

if [[ ! -e "$RYZENAI_EP_PATH" ]]; then
    echo "WARNING: $RYZENAI_EP_PATH not found; the NPU provider will not load" >&2
fi
