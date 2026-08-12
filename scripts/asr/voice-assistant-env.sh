#!/usr/bin/env bash
# Environment for the combined ASR + LLM voice assistant. Source it, do not execute.
#
# Needs BOTH runtimes in one process:
#   - Whisper via the VitisAI ONNX Runtime EP  -> needs the peano library path
#   - Phi-3.5-mini via onnxruntime-genai       -> needs deployment/lib and RYZENAI_EP_PATH
#
# deployment/ lives with the LLM, so LLM_DIR must point at the run_llm directory
# even though the assistant runs from run_asr.

RYZENAI_VENV="${RYZENAI_VENV:-/home/amd/ryzenai/venv}"
LLM_DIR="${LLM_DIR:-/home/amd/run_llm}"

if [[ ! -f "$RYZENAI_VENV/bin/activate" ]]; then
    echo "ERROR: no venv at $RYZENAI_VENV" >&2
    return 1
fi
if [[ ! -d "$LLM_DIR/deployment/lib" ]]; then
    echo "ERROR: no deployment/lib under $LLM_DIR (set LLM_DIR)" >&2
    return 1
fi

# Runtime-managed files intentionally live outside this repository.
# shellcheck disable=SC1091
source "$RYZENAI_VENV/bin/activate"
# shellcheck disable=SC1091
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${RYZEN_AI_INSTALLATION_PATH}/onnxruntime/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${RYZEN_AI_INSTALLATION_PATH}/lib/python3.12/site-packages/lnx64.o/tools/peano/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LLM_DIR/deployment/lib
export RYZENAI_EP_PATH=$LLM_DIR/deployment/lib/libonnxruntime_providers_ryzenai.so
