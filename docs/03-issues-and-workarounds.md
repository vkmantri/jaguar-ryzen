# Ryzen AI 1.8 on Ubuntu 24.04 / Krackan NPU — Issues, and how it was solved

**Platform:** Jaguar P122a devkit — Ryzen Embedded V4A46X, NPU `1022:17f0` rev `0x20`
(`NPU Krackan 2`, npu6, aie2p 6×8), Ubuntu 24.04.4, kernel 7.0.0-28-generic
**Software:** Ryzen AI 1.8.0 Linux, XRT 2.25.37, amdxdna DKMS 0.15


Format follows `github.com/kotetsuy/ryzenai_1_8`, which documented the same software on
Ubuntu 26.04 / Strix Halo. Where our resolution differed, that is called out explicitly.

| Workload | Document |
|---|---|
| LLM — Phi-3.5-mini (OGA flow) | `docs/llm/01-llm-phi35-findings.md` |
| ASR — Whisper (VitisAI EP flow) | `docs/asr/01-asr-whisper-findings.md` |
| Voice agent — ASR + LLM + TTS | `docs/voice-agent/01-voice-agent-findings.md` |

One cross-cutting constraint is worth knowing before combining workloads:
**ONNX Runtime and onnxruntime-genai cannot both load a model in the same process.** Both
statically link protobuf and each registers `external_data.proto`; the second registration is
fatal (`CHECK failed: GeneratedDatabase()->Add(...)`) and the process wedges. Importing both
modules succeeds, so an import-level check gives a false pass — the collision happens at model
load. Use process isolation. Details in the voice agent document, Issue 1.

---

---

## Problem 1 — `Test Finished`, exit code 0, and the NPU was never used

Everything looks like success.

**Symptom** `quicktest.py` prints `Test Finished` and exits 0. Buried at the top of the log:

```
[E:onnxruntime:, provider_bridge_ort.cc:2353 Create] ... Failed to load library
.../libonnxruntime_vitisai_ep.so with error:
libpeano-lib.so.21.0git: cannot open shared object file: No such file or directory
```

**Cause.** `libonnxruntime_vitisai_ep.so` needs `libpeano-lib.so.21.0git`, which ships only in
`site-packages/lnx64.o/tools/peano/lib` — not on the `LD_LIBRARY_PATH` the documentation tells
you to set. When the EP fails to load, ONNX Runtime prints one error, falls back to CPU, and
completes normally.

**Fix.**

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${RYZEN_AI_INSTALLATION_PATH}/lib/python3.12/site-packages/lnx64.o/tools/peano/lib
```

checking `objdump -p libonnxruntime_vitisai_ep.so | grep NEEDED | grep peano` on all four copies of the
library — no match, so it looked inapplicable. That check is invalid: `NEEDED` lists only
direct dependencies, and libpeano is transitive. Only an actual load reveals it.

**How we made it un-repeatable.** We do not use exit codes as a success signal. Every VitisAI
run captures the EP's native output at file-descriptor level and refuses to report numbers
unless NPU execution is proven. A regression test uses the verbatim failure log.

> Differs from kotetsuy: they document the fix and advise grepping the log manually. We made
> a fallback *fail the run* rather than rely on remembering to check.

---

## Problem 2 — `libxrt_core.so.2` "failed to open library" when the file is fine

**Symptom.** EP loads, model compiles, then aborts:

```
Failed to open library '/opt/xilinx/xrt/lib/libxrt_core.so.2'
terminate called after throwing 'vaip_core::GlogFatalException'
```

The file exists, is readable, and `ldd` resolves it.

**Cause.** The venv's `activate` puts `voe/lib` early on `LD_LIBRARY_PATH`. It ships
`libxrt_coreutil.so.2.19.184`, which lacks `xrt_core::smi::get_option_options` present in the
system 2.25.37. The old library wins, so the newer `libxrt_core.so.2` fails to `dlopen` with an
undefined-symbol error that is *reported* as a file-open failure.

**Fix.** Source XRT's setup after activating the venv — it prepends `/opt/xilinx/xrt/lib`:

```bash
source $RYZEN_AI_INSTALLATION_PATH/bin/activate
source /opt/xilinx/xrt/setup.sh
```

Verified ordering: `/opt/xilinx/xrt/lib` at position 3, `voe/lib` at position 6.

 For any "cannot open library", use `LD_DEBUG=libs` to see which file was actually
chosen. `ldd` resolves with its own environment and will mislead you.

---

## Problem 3 — the kernel's in-tree `amdxdna` enumerates the NPU but cannot execute

**Symptom.** With XRT 2.25.37 user space on the in-tree driver:

- `xrt-smi examine` works — device listed, firmware 1.0.0.63 reported
- `xrt-smi validate --run all` — all tests fail with `ERT_CMD_STATE_ABORT`
- kernel log:
  ```
  amdxdna 0000:03:00.1: xdna_mailbox.78: Message callback ret -22
  amdxdna 0000:03:00.1: xdna_mailbox.78: Unexpected ret -22, disable irq
  ```

`-22` is `-EINVAL`: the driver cannot parse the firmware's mailbox responses.

**Cause.** Two coupled version mismatches. In-tree driver is 0.7.0; the DKMS package is 0.15.
Both expose the *same five ioctls* (verified with `nm`), so it is not a missing-ioctl problem.
The in-tree driver also loads production firmware `npu.sbin` (1.0.0.63) while the DKMS package
ships `npu.dev.sbin` (1.1.2.64). Driver and firmware are versioned together.

**Fix.** Install `xrt_plugin...amdxdna.deb`. Its `postinst` runs `rmmod amdxdna`, builds DKMS
0.15, and reloads. After that all three validate tests pass:
GEMM 5.7 TOPS, latency 53.0 µs, throughput 82,310 op/s.


> `dkms.conf` declares `DEST_MODULE_LOCATION=/kernel/extras` but DKMS actually installs to
> `updates/dkms/`. Cleanup scripts targeting the declared path silently do nothing.

Kernel is tainted afterwards — the module is signed with an unenrolled MOK
(`module verification failed: signature and/or required key missing`). Harmless with Secure
Boot off; with Secure Boot on you must enrol the key or the NPU driver will not load.

---

## Problem 4 — the default `optimize_level` makes the NPU slower than the CPU

**Symptom.** Our model compiled for BF16 with the documented default config ran at 2.80 FPS /
352 ms, against 5.69 FPS / 169 ms on CPU. The NPU looked 2× slower.

**Cause.** The documentation's "default configuration file for compiling BF16 models" uses
`"optimize_level": 1`. Valid values are 1, 2, 3.

**Fix.** `"optimize_level": 3` → 17.05 FPS / 57.63 ms, a 6.1× speedup from an identical
source model, turning "2× slower than CPU" into "3× faster".

| BF16 config | FPS | p50 | Compile |
|---|---:|---:|---:|
| `optimize_level 1`, storage `auto` | 2.80 | 352.37 ms | 423 s |
| `optimize_level 1`, storage `vectorized` | 2.80 | 352.33 ms | 425 s |
| `optimize_level 3`, storage `vectorized` | **17.05** | **57.63 ms** | 803 s |

Two secondary findings:

- `preferred_data_storage` made no measurable difference (352.37 vs 352.33 ms) despite the
  docs presenting `vectorized` as the CNN-oriented option. Two independent 7-minute compiles
  produced latency identical to 0.01 %.
- **INT8** has almost no sensitivity to the same knob: `opt_level` 0 → 3 moved 40.47 → 41.84
  FPS (+3 %). Tuning conclusions do not transfer between precisions.

---

## Problem 5 — partitioning markers only appear on a cold compile

**Symptom.** A verified-working configuration reports "no VitisAI EP partitioning markers
found" when re-run. Session creation drops from 423 s to 0.5 s.

**Cause.** Two compounding issues:

1. The markers are logged at INFO; ONNX Runtime defaults to Warning (2). Without
   `session_options.log_severity_level = 1` the log contains only `[W:onnxruntime` lines.
2. Even with INFO enabled, a warm cache compiles nothing and therefore logs nothing.

**Impact.** Log-based verification cannot work for repeat runs — precisely the runs you use for
benchmarking. The temptation is to disable the check, which is how silent CPU fallback gets
back in.

**Fix.** Ask the driver instead, which works regardless of cache state:

```bash
xrt-smi examine --report aie-partitions
```

```
Partition Index : 0    Columns: [0,1,2,3,4,5,6,7]
PID 103599 | Ctx ID 1 | Status Active | Submissions 199 | Completions 198 | GOPS 84
```

Idle reads `No hardware contexts running on device`. Our harness now checks the log first and
falls back to matching its own PID against the driver's active hardware contexts.

---

## Problem 6 (More of a Lesson Learned) — installing AMD Quark breaks ONNX Runtime completely

**Symptom.** After `pip install amd-quark` in the Ryzen AI venv, everything stops:

```
ImportError: import numpy failed
```

**Cause.** Irreconcilable pins:

```
flexml     requires numpy<=1.26.4      # the BF16 compiler, part of Ryzen AI
amd-quark  requires numpy>=2.0
```

Quark upgraded numpy to 2.5.2 and ONNX Runtime's compiled extension could no longer load.


Verify with a real inference, not just an import — BF16 returned to 59.84 ms.

**Fix (correct approach).** Quantization is **offline**. Use a separate venv:

```bash
python3.12 -m venv /opt/quark-venv
/opt/quark-venv/bin/pip install amd-quark onnx onnxruntime
/opt/quark-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`torch` is undocumented but required — Quark JIT-compiles custom ops against it on first use.

**Lesson.** Check dependency constraints before installing into a working environment:
`pip install --dry-run`, or read `importlib.metadata.requires()`.

---

## Problem7 — INT8 leaves two operators on the CPU (this is fine)

**Symptom.** INT8 reports `NPU 484 / VITIS_EP_CPU 2` where BF16 reported zero CPU operators.

**Diagnosis.** Enable the per-operator report:

```bash
export XLNX_ONNX_EP_REPORT_FILE=vitisai_ep_report.json   # plus enable_cache_file_io_in_mem=0
```

```json
{"name": "all",          "nodeNum": 486}
{"name": "NPU",          "nodeNum": 484}
{"name": "VITIS_EP_CPU", "nodeNum": 2,
 "supportedOpType": ["::DequantizeLinear", "::QuantizeLinear"]}
```

**Conclusion.** The two CPU nodes are the graph-boundary conversions — fp32 input quantised on
entry, int8 output dequantised on exit. Every interior operator is on the NPU. Expected, not a
partitioning failure.

---

## Problem 7 — the installer's `/usr/include/asm` advice is wrong on 24.04

**Symptom.** `install_ryzen_ai.sh` ends with:

```
Warning: /usr/include/asm does not exist.
  sudo ln -s /usr/include/asm-generic /usr/include/asm
```

Do not follow this on Ubuntu 24.04 amd64.

**Cause.** Stale Ubuntu 22.04 advice, printed unconditionally. On 24.04 amd64,
`/usr/include/x86_64-linux-gnu/asm` already exists and gcc finds it through the multiarch
include path.

**Verify for yourself:**

```bash
printf '#include <asm/types.h>\nint main(void){return 0;}\n' > /tmp/t.c && gcc -c /tmp/t.c -o /tmp/t.o && echo OK
echo | gcc -E -Wp,-v - 2>&1 | grep x86_64-linux-gnu
```

symlinking `/usr/include/asm` → `asm-generic` shadows
architecture-specific headers with generic ones. The install is fine without it.

---

## Problem 8 — device node permissions

**Symptom.** `PermissionError: [Errno 13]` opening `/dev/accel/accel0`.

**Cause.** The node is `root:render 0660` with **no ACL** — note that `/dev/dri/renderD128` and
`card1` carry a trailing `+` (logind ACL) while `accel0` does not, so a desktop login does not
grant access.

**Fix.** Either install the XRT plugin (its `postinst` writes
`/etc/udev/rules.d/99-amdxdna.rules` with `MODE="0666"`), or preferably use least privilege:

```
KERNEL=="accel*",DRIVERS=="amdxdna",GROUP="render",MODE="0660"
```

plus `sudo usermod -aG render $USER` and a full logout/login (a new terminal in an existing
session inherits the old groups). `sg render -c '...'` works for a one-shot test without
logging out.

---

## iGPU

1. AMD's own Linux documentation rules it out

> "RyzenAI-SW repo hosts a diverse set of examples with both NPU and iGPU Execution Provider.
> However, Linux currently supports NPU only flow.
> — Ryzen AI Linux Installation Instructions

2. The referenced example requires DirectML, which is Windows-only.
`RyzenAI-SW/CNN-examples/iGPU/getting_started` states it uses "Olive to convert the model …
and DirectML execution provider to run the model on the iGPU". DirectML is a DirectX 12
API. There is no Linux implementation.

3. This system has no GPU execution provider available.

| Check | Result |
|---|---|
| iGPU present | `1002:1902` rev e2, driver `amdgpu` (Krackan RDNA 3.5) — working |
| `/opt/rocm` | absent |
| `rocminfo` / `rocm-smi` / `hipcc` | not installed |
| ORT providers (stock venv) | `['AzureExecutionProvider', 'CPUExecutionProvider']` |
| ORT providers (Ryzen AI venv) | `['VitisAIExecutionProvider', 'CPUExecutionProvider']` |

No DirectML, no ROCm, no MIGraphX, no CUDA. The iGPU is functional for graphics — it renders
the desktop and reports 8.04 W idle through `amdgpu` hwmon — but there is no ONNX inference
path to it.

---

## Items to check for NPU execution:

```bash
# 1. Driver is the DKMS build, not in-tree
modinfo amdxdna | grep -E '^(filename|version)'      # expect updates/dkms/, 2.25.x

# 2. Device executes, not just enumerates
xrt-smi validate --run latency                        # must PASS

# 3. EP is actually available
python -c "import onnxruntime as o; print(o.get_available_providers())"

# 4. The model ran on the NPU — cold compile
grep "Actually running on NPU" run.log                # >= 1
grep -cE "\[E:|\[F:" run.log                          # 0

# 5. …or warm cache
xrt-smi examine --report aie-partitions               # status Active for your PID

# 6. Per-operator assignment
XLNX_ONNX_EP_REPORT_FILE=report.json                  # + enable_cache_file_io_in_mem=0
```
