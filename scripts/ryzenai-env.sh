#!/usr/bin/env bash
# Environment for running models on the Ryzen AI NPU. Source it, do not execute:
#   source scripts/ryzenai-env.sh
#
# Order matters and is load-bearing:
#  1. activate the venv first
#  2. then source XRT's setup.sh -- it PREPENDS /opt/xilinx/xrt/lib, which must win
#     over the venv's voe/lib. That directory ships libxrt_coreutil.so.2.19.184,
#     which lacks xrt_core::smi::get_option_options present in the system 2.25.37;
#     if it wins, libxrt_core.so.2 fails to dlopen with a misleading
#     "failed to open library" that is really an undefined-symbol error.
#  3. the peano path is NOT optional. libonnxruntime_vitisai_ep.so pulls in
#     libpeano-lib.so.21.0git *transitively* (it is not a direct NEEDED entry, so
#     objdump -p will not show it). Without this, the EP fails to load and ONNX
#     Runtime SILENTLY FALLS BACK TO CPU while still printing "Test Finished"
#     and exiting 0.

RYZENAI_VENV="${RYZENAI_VENV:-/home/amd/ryzenai/venv}"

if [[ ! -f "$RYZENAI_VENV/bin/activate" ]]; then
    echo "ERROR: no venv at $RYZENAI_VENV (set RYZENAI_VENV)" >&2
    return 1
fi

# Runtime-managed files intentionally live outside this repository.
# shellcheck disable=SC1091
source "$RYZENAI_VENV/bin/activate"
# shellcheck disable=SC1091
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${RYZEN_AI_INSTALLATION_PATH}/onnxruntime/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${RYZEN_AI_INSTALLATION_PATH}/lib/python3.12/site-packages/lnx64.o/tools/peano/lib

if [[ ! -e "${RYZEN_AI_INSTALLATION_PATH}/lib/python3.12/site-packages/lnx64.o/tools/peano/lib/libpeano-lib.so.21.0git" ]]; then
    echo "WARNING: libpeano-lib.so.21.0git not found; the VitisAI EP will fall back to CPU" >&2
fi
