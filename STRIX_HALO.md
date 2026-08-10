# DS4 on Strix Halo — Performance Investigation

## Hardware

- **Machine:** Framework Desktop, AMD Strix Halo APU
- **RAM:** 125 GB (128 GB physical)
- **GPU:** AMD Radeon 8060S (gfx1151, RDNA 3.5), 124 GiB GTT
- **OS:** Fedora 44, Linux 7.1.7
- **SSD:** WD_BLACK SN7100 4TB NVMe (~7 GB/s sequential)

### Kernel Parameters

```
amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

## Model

DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
- 80.76 GiB resident (86.7 GB on disk)
- 1M token max context (284B params, 13B active)

## Benchmark Results (64K context)

See `speed-bench/strix_halo.csv` and `speed-bench/strix_halo_ssd.csv`.

| mode | ctx | prefill t/s | gen t/s | notes |
|---|---|---|---|---|
| resident | 2K–64K | 157–212 | 13.1–15.8 | fast, interactive |
| SSD streaming | 2K–64K | 50–59 | 9.9–12.0 | expert cache 69 GiB auto |

### Resident Sweep (64K CSV)

- Model loads in ~18 s (18 GiB/s effective into GTT)
- Memory: 81–82 GiB planned (model 80.76, KV 0.4–1.2, buffers 0.03–0.5)
- Prefill: 188–212 t/s at 2K → 157 t/s at 64K
- Generation: 15.8 t/s at 2K → 13.1 t/s at 64K
- KV cache: 52 MB at 2K → 898 MB at 64K

## The 64K Context Cliff

The DeepSeek RoPE original context is **65,536**. Above this threshold, the model switches from native RoPE to YaRN frequency scaling. On the ROCm backend, this causes a catastrophic performance collapse:

| ctx | compressed KV rows | mechanism | 2048-token prefill |
|---|---|---|---|
| 64K | 16,418 | RoPE (native) | **11 s (~188 t/s)** |
| 66K | 16,801 | YaRN scaling | **>60 s (<34 t/s)** |
| 200K | 50,000 | YaRN scaling | **>8 min (~0.2 t/s)** |
| 384K | 98,306 | YaRN scaling | **>11 min (~0.1 t/s)** |
| 1M | 262,146 | YaRN + managed KV | **>45 min (never finished)** |

### Root Causes

1. **GPU attention at large compressed-KV dimensions:** hipBLAS attention GEMMs degrade superlinearly at row counts above ~16K. At 98K rows (384K ctx), the throughput is ~72× worse than linear scaling predicts.

2. **Managed KV fallback at large contexts:** When `context_bytes ≥ 8 GiB` and device free memory is tight, the allocator falls back to `hipMallocManaged` (host memory). On resident mode (model 80.76 GiB in GTT), free memory is always tight, so managed KV triggers at ~370K+ ctx.

3. **SSD streaming arena exhaustion:** The stream-span staging arena is capped at ~48 GiB. When the expert cache ring plus span staging exceeds 124 GiB, `cudaMalloc` fails.

### Code Fixes Applied (commit `0bccfdd`)

| fix | file | change |
|---|---|---|
| Managed KV threshold | `rocm/ds4_rocm_runtime.cuh`, `ds4_cuda.cu` | Remove hard 8 GiB cutoff; always consult free memory |
| Arena span limit | `rocm/ds4_rocm_runtime.cuh` | `total/3` → `total*3/4` to prevent premature OOM |
| Managed KV reserve | `rocm/ds4_rocm_runtime.cuh`, `ds4_cuda.cu` | `total/4` → `total/8` to keep KV device-resident when possible |
| Makefile dependency | `Makefile` | `ds4_rocm.o` already tracks `$(ROCM_SRCS)`; `-B` flag on `strix-halo` target forces rebuild |

## SSD Streaming at 512K+ Contexts

Extensive testing of SSD streaming with various expert cache sizes (8–69 GiB) at 384K–1M ctx:

| cache | ctx | planned GiB | result |
|---|---|---|---|
| auto (69 GiB) | 512K | 84.6 | arena OOM |
| auto (69 GiB) | 1M | 95.4 | arena OOM |
| 67 GB | 512K | 79.1 | functional but 26–48× read amplification, >15 min per frontier |
| 56 GB | 1M | 78.8 | functional but 75× read amplification, never finished |
| 48 GB | 1M | 74.8 | functional but 170–415× read amplification, never finished |

The read amplification is caused by the q8-fp16 upcast cache being starved (16 GiB streaming reserve) and the expert cache ring not holding the full expert set (needs ~72 GiB for 11,008 experts).

## GLM 5.2 SSD Streaming

GLM 5.2 UD-Q2_K_RoutedQ2K (262 GB model) with `--ssd-streaming` on Strix Halo was tested and found infeasible:

- Auto budget capped at 67.67 GiB cache (5,354 of 7,000+ experts) by memory guard for ctx=4096
- 0 full resident layers (auto)
- 1.47 t/s generation, 12.7 t/s prefill
- Read amplification 2.4× and climbing after 2 frontiers
- Frontier 2 never completed in 12+ minutes

## Performance Recommendations

### What Works Well (Interactive)

- **Resident mode at ≤64K ctx** — 157–212 t/s prefill, 13–16 t/s gen
- **SSD streaming at ≤64K ctx** — 50–59 t/s prefill, 10–12 t/s gen

### What Needs Code Improvements

- **GPU attention at >64K ctx:** The hipBLAS indexed-prefill attention GEMM has a superlinear performance cliff above ~16K compressed rows. A Vulkan attention kernel or FlashAttention-style tiled implementation could avoid this.
- **SSD streaming span staging:** The arena accumulates spans across layers during prefill instead of releasing per-layer. Peak accumulation (~37–48 GiB) forces the expert cache ring below the full-expert-set size, causing churn.
- **q8-fp16 cache starvation:** Streaming free reserve (16 GiB) prevents the upcast cache from engaging, forcing slow q8 kernels.

## Environment Variables (Strix Halo ROCm)

| variable | default | range | effect |
|---|---|---|---|
| `DS4_ROCM_STREAM_MODEL_CACHE_GB` | total/3 (41 GiB) | 8–128 | arena limit for stream spans |
| `DS4_ROCM_STREAM_FREE_RESERVE_GB` | 16 | 2–64 | headroom kept free in streaming |
| `DS4_ROCM_STREAM_Q8_F16_CACHE_GB` | total/8 (15.5) | 0–128 | q8→fp16 upcast cache limit (0 = disable) |
| `DS4_BATCHED_ROPE_MAX` | 4096 | 0–65536 | max tokens for batched CPU rope |
| `DS4_PARALLEL_ATTN_ROWS` | (unset) | — | enable parallel attention rows at pos0=0 |

## Commits on `fedora44-ds4`

```
0bccfdd rocm: improve managed-KV threshold, arena limit, and reserve for large-context streaming
d897da4 bench: add AMD Strix Halo (Radeon 8060S) resident and SSD streaming results
de65cae makefile adjustments for fedora44 build
```
