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

## The 64K Context Cliff

The DeepSeek RoPE original context is **65,536**. Above this, the model switches
to YaRN frequency scaling on ROCm, causing catastrophic prefill collapse:

| ctx | compressed KV rows | mechanism | 2048-token prefill |
|---|---|---|---|
| 64K | 16,418 | RoPE (native) | **~11 s** (~188 t/s) |
| 66K | 16,801 | YaRN scaling | **>60 s** (<34 t/s) |
| 200K | ~50,000 | YaRN scaling | **>8 min** (~0.2 t/s) |
| 384K | 98,306 | YaRN scaling | **>11 min** (~0.1 t/s) |
| 1M | 262,146 | YaRN + managed KV | **>45 min** (never finished) |

**Root cause:** The prefill attention uses `cublasGemmStridedBatchedEx`
(hipBLAS/rocBLAS) with `CUBLAS_GEMM_DEFAULT`. At compressed-KV rows above
~16K, rocBLAS picks a suboptimal tiling strategy and throughput collapses.

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

## Performance Fixes for >64K Contexts

### rocBLAS / hipBLAS Tuning (no code changes)

System-level env vars — may or may not help depending on whether gfx1151
has pre-tuned kernels in the tensile library:

| variable | effect |
|---|---|
| `ROCBLAS_TENSORLIB_PATH` | path to tensile library YAML — enables per-shape autotuning |
| `HIPBLASLT_TENSORLIB_PATH` | same for hipBLASLt |
| `ROCBLAS_GEMM_ALGO_ALWAYS=1` | force rocBLAS to always query tensile |

Code-level tuning (ds4 changes):

1. `rocm/ds4_rocm_hipblaslt.cuh`: set `MAX_WORKSPACE_BYTES` to 64 MiB
   so the heuristic considers scratch-space algorithms for large GEMMs.
2. Try all 8 candidates from `hipblasLtMatmulAlgoGetHeuristic`, not just `heur[0]`.
3. `rocm/ds4_rocm_runtime.cuh`: replace `CUBLAS_GEMM_DEFAULT` with an explicit
   algorithm search when compressed rows exceed 16K.

### FlashAttention via HIP Kernel

The indexed-prefill attention shape:
`Q[n_tok×128, 64] @ K^T[64, n_kv_rows×128]`. At 384K ctx, the attention matrix
`S` is 805 MB per head. FA avoids materializing `S` by streaming K/V blocks
through LDS with online softmax.

**Prerequisites:**
- rocWMMA installed per `STRIX_HALO.md` (warp-level MMA for gfx1151)
- HIP LDS: 64 KB per CU on gfx1151 — constrains tile sizes

**Kernel sketch:**

```
Input:  Q [n_tok, n_heads, head_dim]       e.g. [2048, 128, 64]
        K [n_kv_rows, n_heads, head_dim]   e.g. [98306, 128, 64]
        V [n_kv_rows, n_heads, head_dim]
Output: O [n_tok, n_heads, head_dim]

Block:  Br=64 tokens, 1 head
        Qi_tile[Br, 64] in LDS (8 KB)
        Oi[Br, 64], mi[Br], li[Br] in registers

Loop over K/V streamed in Bc=256-row tiles (32 KB LDS + Qi):
  Sij[Br, Bc] = Qi_tile @ Kj_tile^T     (rocWMMA)
  mij = rowmax(Sij), lij = rowsum(exp(Sij - mij))
  Oi = diag(exp(mi - mij)) * Oi + exp(Sij - mij) @ Vj
  mi = max(mi, mij); li = exp(mi - mij) * li + lij

Output: O[block] = diag(1/li) * Oi
```

**Tile sizes (64 KB LDS):** Br = 64 (8 KB Q tile), Bc_sub = 256 (32 KB K tile).

**Memory:** O(n_tok × n_kv) global reads (K/V streamed), O(n_tok × heads × dim)
writes (O once). No S matrix materialization.

**Integration points:**
- Replace `cublasGemmStridedBatchedEx` / `hipblasLtMatmul` when
  `n_kv_rows > 16384` and `backend == rocm`
- Reuse existing `cuda_tmp_alloc()`, stream management
- Template in `rocm/ds4_rocm_runtime.cuh` alongside existing GLM prefill kernels

### Env Vars (Strix Halo ROCm)

| variable | default | effect |
|---|---|---|
| `DS4_ROCM_STREAM_MODEL_CACHE_GB` | total/3 | arena limit for stream spans |
| `DS4_ROCM_STREAM_FREE_RESERVE_GB` | 16 | headroom in streaming mode |
| `DS4_ROCM_STREAM_Q8_F16_CACHE_GB` | total/8 | q8→fp16 upcast cache limit |
| `DS4_BATCHED_ROPE_MAX` | 4096 | max tokens for batched CPU rope |
| `DS4_PREFILL_PROFILE_DETAIL` | (unset) | time breakdown (DeepSeek — unimplemented) |
| `DS4_PREFILL_CHUNK` | 4096 | tokens per GPU prefill chunk |

## Validation Plan (for FlashAttention)

1. **Correctness:** Compare FA output against GEMM path for all frontier
   ctx values 2K–64K. Tolerance 1e-3 on logits (FP16 accumulation differs).
2. **Prefill sweep at the cliff:** Run `ds4-bench --gen-tokens 0` at 65536,
   131072, 262144, 393216. Verify prefill t/s recovers to within ~10× of 64K
   (vs. current ~72× degradation). No OOM, no NaNs.
3. **Generation consistency:** Run `--gen-tokens 128` at 64K and 128K.
   KV cache bit-exact vs. GEMM path; gen t/s unchanged (FA prefill-only).
4. **Long-running smoke:** Continuous 2K→128K growth with `--gen-tokens 16`
   per frontier, verifying no cache drift over 100+ steps.
5. **Benchmark:** Produce `speed-bench/strix_halo_fa.csv` + SVG, 2K→393K,
   comparable to existing resident sweep.

## Commits on `fedora44-ds4`

```
bb95f73 docs: add Strix Halo performance investigation
0bccfdd rocm: improve managed-KV threshold, arena limit, and reserve
d897da4 bench: add AMD Strix Halo resident and SSD streaming results
de65cae makefile adjustments for fedora44 build
```
