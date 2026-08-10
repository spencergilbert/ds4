# Strix Halo Performance Investigation

Hardware: Framework Desktop, AMD Strix Halo APU, 125 GB RAM, Radeon 8060S (gfx1151, RDNA 3.5),
124 GiB GTT, WD_BLACK SN7100 4TB NVMe. Fedora 44, Linux 7.1.7.

## Benchmark Results (64K Context)

Model: DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
(80.76 GiB resident, 86.7 GB on disk, 1M max context).

See `speed-bench/strix_halo.csv` and `speed-bench/strix_halo_ssd.csv`.

### Resident (model fully in GTT)

| ctx | prefill t/s | gen t/s | gen first ms | planned GiB |
|---|---|---|---|---|
| 2K | 188.8 | 15.8 | 63 | 81.2 |
| 16K | 197.5 | 14.4 | 70 | — |
| 32K | 182.7 | 13.8 | 72 | — |
| 64K | 157.5 | 13.1 | 76 | 82.5 |

- Model loads in ~18 s
- Gen degrades gently: 15.8 → 13.1 t/s over the sweep
- Prefill: 157–212 t/s

### SSD Streaming (69 GiB auto expert cache, 0.99 GiB resident)

| ctx | prefill t/s | gen t/s | planned GiB |
|---|---|---|---|
| 2K | 50.4 | 9.9 | 74.0 |
| 64K | 55.0 | 9.5 | — |

- Prefill 3–4× slower than resident (expert loading from SSD)
- Gen ~20–25% slower (most experts stay in cache after first pass)

## The 64K Context Cliff (Revised 2026-08-10)

**Update:** the earlier "72x attention cliff" does not reproduce on the current
binary. DeepSeek V4 Flash uses a compressed-KV indexer: once `n_comp > 512`
(`DS4_N_INDEXER_TOP_K`), prefill attention runs the **indexed online kernel**
(fixed top-k 512 + 128-token raw window), whose cost is flat in context. The
`cublasGemmStridedBatchedEx` S-matrix path only runs for `n_comp <= 512`.

The real context-scaling cost is the **indexer scores GEMM** (WMMA128 kernel,
O(n_comp) per chunk) plus indexer top-k. Measured per 4096-token layer/chunk:

| ctx | compressed KV rows | indexer score | indexed attention | prefill t/s (full) |
|---|---|---|---|---|
| 64K | 16,386 | ~181 ms | ~108 ms | 192.5 |
| 128K | 32,770 | ~360 ms | ~108 ms | 163.2 |
| 200K | 49,154 | ~540 ms | ~108 ms | 142.7 |
| 384K | 98,306 | ~1.1 s | ~108 ms | 102.0 |

Prefill degrades linearly (indexer scores dominate), not catastrophically.
384K is ~1.9x slower than 64K, not 72x.

The indexer scores WMMA128 kernel was rewritten (32-token blocks, register-
resident weighted-ReLU reduction with the measured rocwmma accumulator layout,
double-buffered a_sh): 182.6 → 129.3 ms at 16384 comps x 4096 tokens,
**bit-identical logits**. End-to-end: 64K 192.5 → 198.8 t/s, 128K 163.2 →
171.3 t/s.

## Code Fixes Applied

See commit `0bccfdd` on branch `fedora44-ds4`.

| fix | file | change |
|---|---|---|
| Managed KV threshold | `rocm/ds4_rocm_runtime.cuh`, `ds4_cuda.cu` | Remove hard 8 GiB cutoff; always consult free memory |
| Arena span limit | `rocm/ds4_rocm_runtime.cuh` | `total/3` → `total*3/4` for SSD streaming headroom |
| Managed KV reserve | `rocm/ds4_rocm_runtime.cuh`, `ds4_cuda.cu` | `total/4` → `total/8` so more contexts keep device KV |

## SSD Streaming at Large Contexts

Extensive testing with various expert cache sizes (8–69 GiB) at 384K–1M ctx:

| cache | ctx | planned GiB | result |
|---|---|---|---|
| auto (69 GiB) | 512K | 84.6 | arena OOM |
| auto (69 GiB) | 1M | 95.4 | arena OOM |
| 67 GB | 512K | 79.1 | functional, 26–48× read amplification, >15 min/frontier |
| 56 GB | 1M | 78.8 | functional, 75× read amplification, never finished |

The read amplification is from the q8-fp16 upcast cache being starved
(16 GiB streaming reserve) and the ring not holding the full 11K-expert set.

## GLM 5.2 SSD Streaming

GLM 5.2 UD-Q2_K_RoutedQ2K (262 GB) tested and found infeasible:
- Auto cache capped at 67.67 GiB (5,354 experts) by memory guard at ctx=4096
- 0 full resident layers (auto)
- 1.47 t/s gen, 12.7 t/s prefill
- Read amplification 2.4× and climbing after frontier 1; frontier 2 never finished

## Performance Fixes for >64K Contexts (Revised 2026-08-10)

### Status of the old rocBLAS / hipBLAS tuning plan — tested, no effect

- `ROCBLAS_GEMM_ALGO_ALWAYS=1`: 191.3 vs 192.6 t/s baseline — noise.
- `ROCBLAS_TENSORLIB_PATH` / `HIPBLASLT_TENSORLIB_PATH`: gfx1151 tensile
  libraries exist (`/usr/lib64/rocblas/library/`, `/usr/lib64/hipblaslt/library/`)
  but there are no per-shape YAML autotune files to point at.
- The hipBLASLt plan machinery (`rocm/ds4_rocm_hipblaslt.cuh`,
  `MAX_WORKSPACE_BYTES`, the 8 heuristic candidates) is **dead code** on this
  branch — `hipblaslt_gemm_tn_f16_out_f16` has no callers; fp16 GEMMs use
  `cublasGemmEx(..., CUBLAS_GEMM_DEFAULT)` → rocBLAS directly.
- The proposed `CUBLAS_GEMM_DEFAULT` replacement targets the S-matrix
  attention path, which the indexer model only uses for `n_comp <= 512`.

### The real fix: indexer scores WMMA128 kernel (`rocm/ds4_rocm_indexer.cuh`)

The prefill indexer scores GEMM (n_comp x n_tokens x 64 heads x 128 dims,
weighted ReLU per head) is the dominant context-scaling cost. Rewritten:
- 32 tokens per block (was 16) — halves per-layer q re-reads and block count
- Register-resident weighted-ReLU head reduction (no c_sh round trip), using
  the measured rocwmma gfx1151 accumulator layout: element i of lane l holds
  (row = 2*i + (l>>4), col = l & 15) — this differs from nvcuda
- Double-buffered a_sh, 1 barrier per head (was 3)

Micro-benchmark at 16384 comps x 4096 tokens: 182.6 ms → 129.3 ms
(6.0 → 8.5 TFLOPS), **bit-identical** outputs. End-to-end: 64K 192.5 →
198.8 t/s, 128K 163.2 → 171.3 t/s. 384K re-run pending.

Remaining ideas: indexer top-k pass (~26 ms/layer at 64K, scales with
n_comp), fp16 q/index_comp pre-pass (halves the ~16 GB/layer q re-read
memory traffic, bit-identical MMA inputs).

**Update 2026-08-10:** both landed. Top-k now uses a CUB radix tree over
8192-row chunks (160 → 115 ms at 98K comps, bit-exact); the indexer q is
converted to fp16 in the QAT step (score stage 120 → 113 ms at 64K,
bit-identical). The index_comp (k) side stays fp32 — converting it needs a
cache-format change across ~15 consumer sites for ~2-4% more.

### FlashAttention via HIP Kernel — not needed

Attention is already flat in context via the indexed online kernel
(`attention_indexed_mixed_heads8_online_kernel`, fixed top-k 512 + raw
window). There is no attention cliff to fix; skip FA unless the indexed
attention itself becomes the bottleneck (it is ~108 ms/layer at any ctx,
vs ~1.1 s for indexer scores at 384K).

**Prerequisites:**
- rocWMMA installed per `STRIX_HALO.md` (warp-level MMA for gfx1151)
- HIP LDS: 64 KB per CU on gfx1151 — constrains tile sizes

### Env Vars (Strix Halo ROCm)

| variable | default | effect |
|---|---|---|
| `DS4_ROCM_STREAM_MODEL_CACHE_GB` | total/3 | arena limit for stream spans |
| `DS4_ROCM_STREAM_FREE_RESERVE_GB` | 16 | headroom in streaming mode |
| `DS4_ROCM_STREAM_Q8_F16_CACHE_GB` | total/8 | q8→fp16 upcast cache limit |
| `DS4_BATCHED_ROPE_MAX` | 4096 | max tokens for batched CPU rope |
| `DS4_PREFILL_PROFILE_DETAIL` | (unset) | time breakdown (DeepSeek — unimplemented) |
| `DS4_PREFILL_CHUNK` | 4096 | tokens per GPU prefill chunk |
| `DS4_ROCM_INDEXER_STAGE_PROFILE` | (unset) | per-layer indexer score/topk/attention timing |
| `DS4_ROCM_LAYER_STAGE_PROFILE[_LAYER]` | (unset) | per-layer prefill stage timing |

## Validation Plan (indexer kernel rewrite)

1. **Correctness:** logit dump (`--dump-frontier-logits-dir`) at 8K/64K must be
   **bit-identical** to the old kernel (the rewrite preserves the exact MMA
   math and accumulate order; verified 2026-08-10).
2. **Prefill sweep:** `ds4-bench --gen-tokens 0` at 65536, 131072, 262144,
   393216 — baseline was 192.5 / 163.2 / 142.7 / 102.0 t/s; the rewrite
   measured 198.8 / 171.3 at 64K/128K.
3. **Generation consistency:** `--gen-tokens 128` at 64K and 128K — the
   indexer only feeds top-k selection; KV cache and gen t/s must be unchanged.
4. **Long-running smoke:** continuous 2K→128K growth with `--gen-tokens 16`
   per frontier, verifying no top-k drift.
5. **Benchmark:** produce `speed-bench/strix_halo_fa.csv` + SVG, 2K→393K
   (rename to `strix_halo_idx.csv` if FA is dropped).

## Commits on `fedora44-ds4`

```
fd054d2 rocm: fp16 indexer q for prefill scores (bit-exact, ~6% score stage)
ac9c87a rocm: indexer top-k via CUB radix tree (1.3-1.4x, bit-exact)
3f74827 rocm: rewrite indexer scores WMMA128 kernel (1.47x, bit-exact)
bb95f73 docs: add Strix Halo performance investigation
0bccfdd rocm: improve managed-KV threshold, arena limit, and reserve
d897da4 bench: add AMD Strix Halo resident and SSD streaming results
de65cae makefile adjustments for fedora44 build
```
