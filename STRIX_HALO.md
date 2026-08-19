# DS4 on Strix Halo — Setup Guide

Minimal setup for DS4 ROCm inference on a Strix Halo machine with 128 GB RAM
and Radeon 8060S (`gfx1151`).

## 1. Install ROCm

On Ubuntu 26.04 LTS (also works on Fedora 44 with equivalent packages):

```sh
sudo apt-get update
sudo apt-get install -y \
  hipcc rocminfo rocm-smi \
  libamdhip64-dev \
  libhipblas-dev libhipblaslt-dev \
  librocblas-dev \
  librocwmma-dev \
  libhipcub-dev
```

The backend uses rocWMMA. Install complete matching rocWMMA header tree:

```sh
git clone --depth 1 --branch rocm-7.1.0 https://github.com/ROCm/rocWMMA.git /tmp/rocWMMA-rocm-7.1.0
sudo mkdir -p /usr/local/include
sudo cp -a /tmp/rocWMMA-rocm-7.1.0/library/include/rocwmma /usr/local/include/
```

If ROCm is installed under `/usr` but tooling expects `/opt/rocm`:

```sh
sudo mkdir -p /opt/rocm/bin
sudo ln -sf /usr/bin/hipcc /opt/rocm/bin/hipcc
sudo ln -sfn /usr/lib/x86_64-linux-gnu /opt/rocm/lib
sudo ln -sfn /usr/include /opt/rocm/include
```

## 2. Enable ROCm access

```sh
sudo usermod -aG render,video "$USER"
```

Log out and back in, or reboot. Verify:

```sh
rocminfo | grep -A80 'Name:                    gfx1151'
```

## 3. Increase GPU-visible memory

Kernel parameters:

```text
amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

On Ubuntu with GRUB:

```sh
sudo cp /etc/default/grub /etc/default/grub.bak
sudoedit /etc/default/grub
# Set:
# GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856"
sudo update-grub
sudo reboot
```

Verify after reboot:

```sh
cat /proc/cmdline
sudo dmesg | grep -Ei 'GTT|gttsize|TTM|VRAM'
rocminfo | grep -A80 'Name:                    gfx1151'
```

Expected:

```text
amdgpu:  126976M of GTT memory ready
rocminfo gfx1151 pool: 130023424 KB    (~124 GiB)
```

## 4. Build DS4

```sh
make strix-halo -j"$(nproc)"
```

`make rocm` is an alias for `make strix-halo`.

## 5. Use the right GGUF

Use the IQ2XXS/Q2K/Q8 imatrix GGUF:

```text
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

Avoid mixed IQ2/IQ4 or IQ2/Q4 GGUFs on this machine — they put more memory
pressure on the ROCm path and can trigger system OOM.

Optional DSpark speculative decoding (see §6) also needs its own ~5.6 GiB
support GGUF, downloaded once:

```sh
DS4_GGUF_DIR=~/.cache/ds4/models ./download_model.sh ds4f-dspark
```

## 6. Run DS4 optimally (current best-known settings)

Every prefill/decode optimization ships **on by default** — no flags or env
vars are needed for the fast path:

- **Adaptive prefill chunk** from the context size: 16384 tokens ≤ 256K,
  8192 for 256K–768K, 4096 ≥ 768K (the longer the prompt, the bigger the
  chunk, up to the memory budget).
- **fp16 indexer-scores buffer** (halves the top-k score reads) and the
  **MoE WMMA 8-warp hotlist** — always on.
- **WMMA per-token decode score** (3× the scalar indexer score) — always on.
- **WMMA skinny HC projection** (the `mix_hc`=24 GEMM, 1.6×) — always on.
- **Multi-stream batched decode** — on when the server runs with
  `--batched-session N` (per-session streams overlap the latency-bound M=1
  kernels across concurrent sessions).

Measured throughput on this machine (prefill t/s): 4112 → 245, 64K → 250,
128K → 217, 384K → 149; decode ≈ 14.8 t/s single-session, ≈ 14.9 t/s
aggregate with four concurrent sessions.

The ROCm backend is selected automatically; the model path is
`~/.cache/ds4/models/`.

### Interactive / one-shot

```sh
./ds4 -m ~/.cache/ds4/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
```

### DSpark speculative decoding (optional)

DSpark drafts up to five future tokens with a small auxiliary model and
verifies them against the main model's hidden states (the main model stays
authoritative). Opt-in and experimental; it does not accelerate prefill.

```sh
./ds4 --rocm --dspark \
  -m ~/.cache/ds4/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --mtp ~/.cache/ds4/models/DeepSeek-V4-Flash-DSpark-support-0731.gguf
```

| flag | effect |
|---|---|
| `--mtp FILE` | load the support GGUF (DSpark or legacy MTP); required |
| `--dspark` | enable the DSpark runtime |
| `--dspark-confidence F` | enable DSpark with a confidence-pruning threshold 0..1 (implies `--dspark`); default 0.7 on ROCm/CUDA |
| `--dspark-strict` | load the support model but keep target-only decode (reproducibility checks) |

The win is prompt- and length-dependent. On this machine, short greedy runs
measured *slower* than ordinary decode (code: 14.8 vs 17.2 t/s; a one-word
factual prompt: 10.4 vs 17.0 t/s at ctx 32768) — the draft + verification
overhead only amortizes on longer, highly predictable continuations.
Benchmark your own workload before enabling it; `--temp 0` is only for
verifying that DSpark preserves greedy output. The support model adds
~5.6 GiB resident memory and is checkpoint-specific — pair it only with the
0731 Flash model, never DeepSeek V4 PRO.

### Server (OpenAI-compatible API, batched decode)

```sh
./ds4-server --rocm \
  -m ~/.cache/ds4/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --ctx 131072 \
  --batched-session 4 \
  --host 127.0.0.1 --port 8000
```

`--batched-session N` keeps N resident sessions and batches decode-ready
requests across them on separate streams (the multi-stream overlap). N=4 is
the practical ceiling before per-session memory OOMs — use the largest N the
context budget allows.

### Benchmark / regression check

```sh
rm -f /tmp/ds4.lock   # clear a stale lock from a killed run
./ds4-bench -m ~/.cache/ds4/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --prompt-file /tmp/promessi_x6.txt \
  --ctx-start 65536 --ctx-max 65536 --ctx-alloc 65538 --gen-tokens 0
```

(Do not run two ds4 processes concurrently — there is a single-instance
lock; kill a hung run with `pkill -9 -f ds4-bench; rm -f /tmp/ds4.lock`.)

### Run as a service (systemd)

A user-service template + installer are in `systemd/`. Install and start:

```sh
./systemd/install.sh
```

That rewrites the placeholders in `systemd/ds4-server.service` with your repo
path and model (defaults: `--ctx 131072 --batched-session 4`, the
`~/.cache/ds4/models/...` model) and installs it to
`~/.config/systemd/user/ds4-server.service`, with `Restart=on-failure`,
a stale-lock cleanup, and a 120G cgroup cap. Override defaults with env vars:

```sh
DS4_CTX=65536 DS4_BATCH=2 DS4_MODEL=/path/to/model.gguf ./systemd/install.sh
INSTALL_ONLY=1 ./systemd/install.sh   # copy + daemon-reload, don't start
```

Enable lingering so it survives logout (`loginctl enable-linger "$USER"`), and
inspect with `systemctl --user status ds4-server.service` / `journalctl --user
-u ds4-server`.

## 7. Memory / throughput tradeoff knobs

Only flip these if a specific long-context session needs the extra headroom —
each has a measured prefill cost:

| env var | effect | cost |
|---|---|---|
| `DS4_GPU_ATTN_COMP_CACHE_F16=1` | fp16 compressed-KV cache: −5.25 GiB @1M | ~−6% prefill |
| `DS4_ROCM_Q8_F16_CACHE_GB=N` | cap the q8→fp16 cache (default unlimited ~10.6 GiB) | ~5.8% prefill per GiB yielded |
| `DS4_CUDA_SESSION_BATCH_MULTI_STREAM=0` | disable the batched-decode stream overlap | −12% concurrent throughput |
| `DS4_CUDA_SESSION_BATCH_SINGLE_GPU=1` | opt-in FFN M=N grouping | net loss at N≤16 — leave off |

The 1M-token context (pc=4096) fits with no flags (~119.3 GiB used, ~4.7 GiB
free); the 8192-token chunk at 1M does not fit. The `DS4_GPU_ATTN_COMP_CACHE_F16=1`
flag is the cheapest per-GiB lever if a 1M@8192 session is ever required
(net-zero with the 8192-chunk gain).
