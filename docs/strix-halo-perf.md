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

**Final (2026-08-10, after top-k CUB tree, fp16 indexer q, and the 8192-token
prefill chunk default):**

| ctx | original | final | Δ |
|---|---|---|---|
| 64K | 192.6 | 208.2 | +8.1% |
| 128K | 163.2 | 179.9 | +10.2% |
| 384K | 102.0 | 121.5 | +19.1% (64 → 54 min) |

## Remaining Cost Profile (8192-token chunks, layer 0 at 64K tail)

**Update 2026-08-11 (indexer direct-load kernel):** the score stage at the
64K tail dropped 181 → 61 ms/layer; the balance is now top-k 49 ms and
indexed attention 216 ms (flat in ctx), so the score kernel is no longer the
long-context bottleneck. See the indexer kernel section below for the new
prefill sweep.

| stage | ms/8192-token layer | share |
|---|---|---|
| routed MoE (iq2/q2k kernels) | ~298 | ~43% |
| attention output projection | ~107 | ~16% |
| Q path | ~81 | ~12% |
| attention (non-indexer layers) | ~72 | ~10% |
| shared experts, HC, router, ... | ~127 | ~19% |

MoE is ~3.3 TFLOP per 8192-token layer at ~11 TFLOPS effective (2-bit
weights, custom dequant-in-kernel). Indexer scores + top-k + indexed
attention together are ~2.4x cheaper than the MoE after the rewrites.

**Update 2026-08-11 (`a8a74d7`):** the MoE is split into WMMA hot-expert
kernels + scalar cold-expert kernels (hot threshold = 8 pairs per expert,
so small agentic turns are scalar-dominated at only ~1.8-2.6 TFLOPS).
Register-resident epilogues on both WMMA hotlist kernels (no fp32 C-tile
shared round trip; gfx1151 accumulator layout) and a shared-free scalar
gate (xq via L2, row span 1024→256) are bit-exact and cut the MoE:
- gate/up IQ2 WMMA 166 → 133 ms/layer
- down Q2K WMMA 90 → 76 ms/layer
- scalar gate 22.7 → 15.3 ms/layer at 200 tokens
End-to-end prefill +5-6% at 2K-64K (195.8→206.5 at 2K, 236.5→251.3 at 8K,
205.3→218.5 at 64K), +2.6-3% at 128K-384K; decode unchanged.

## Indexer Scores Kernel: Direct-Load (2026-08-11)

`indexer_scores_wmma128_kernel_t` was issue/latency-bound at ~8.5 TFLOPS:
**every block staged the fp16 q through `a_sh` shared memory and hit 64
per-head `__syncthreads()` barriers**. Replaced with a direct-load kernel
(`indexer_scores_wmma128_direct_kernel` in `rocm/ds4_rocm_indexer.cuh`):

- a-fragments load straight from the global fp16 q (row stride
  `n_head*head_dim`) — no a_sh staging, **zero barriers** in the head loop
- two heads interleaved per iteration (4 accumulator pairs in flight); the
  4-head variant spills registers and is 10x slower
- weights preloaded to shared once; `b_sh` (fp16 index_comp) unchanged
- causal mask, fully-masked early-out, and accumulation order unchanged

Measured (micro-bench, `98304 comps x 8192 tokens`): 1538 → 421 ms
(8.6 → 31.3 TFLOPS; MMA-only ceiling ~43 TFLOPS). Breakdown experiments:
barriers cost ~9%, the ReLU epilogue ~7%, the a_sh staging + barrier
serialization ~55% — removing the staging is the win. **Bit-identical**
logits at 16K (including a 16-token partial-tile tail chunk) and 64K
(0/129280 diffs). The direct kernel handles all n_tokens > 1 (partial tiles
read in-bounds of the pc-sized q buffer; out-of-range rows are dropped by
the token guards; weights loads are clamped); the fp32-q staged kernel
remains as the fallback.

End-to-end prefill (8192-token chunks, full-prompt):

| ctx | before | after | Δ |
|---|---|---|---|
| 8K | 251.2 | 252.7 | +0.6% |
| 32K | 237.3 | 243.0 | +2.4% |
| 64K | 219.3 | 229.4 | +4.6% |
| 128K | 185.5 | 201.0 | +8.3% |
| 384K | 124.7 | 142.7 | +14.5% (54 → 46 min) |

Decode unchanged (~13 t/s). The remaining long-context indexer costs are
the top-k CUB tree (49 ms at 64K tail, ~230 ms at 384K) and the indexed
attention (216 ms per 8192-token chunk, flat in ctx) — both now larger
than the score stage. CSV/SVG: `speed-bench/strix_halo_idx_direct.csv`; the
WMMA two-pass attention numbers are in `speed-bench/strix_halo_wmma.csv`
(§"WMMA two-pass indexed attention").

## Code Fixes Applied

See commit `0bccfdd` on branch `fedora44-ds4`.

| fix | file | change |
|---|---|---|
| Managed KV threshold | `rocm/ds4_rocm_runtime.cuh`, `ds4_cuda.cu` | Remove hard 8 GiB cutoff; always consult free memory |
| Arena span limit | `rocm/ds4_rocm_runtime.cuh` | `total/3` → `total*3/4` for SSD streaming headroom |
| Managed KV reserve | `rocm/ds4_rocm_runtime.cuh`, `ds4_cuda.cu` | `total/4` → `total/8` so more contexts keep device KV |

## Max Usable Context: 512K → 1M (2026-08-11)

The resident model (80.76 GiB) plus session buffers used to cap at ~512K
ctx: 768K/1M OOM'd at session create with 14 `ROCm tensor alloc failed`
errors. The planned-memory print (92.47 GiB at 384K) understated the real
usage — measured with `DS4_METAL_MEMORY_REPORT=1` (now printed at session
create too), the model load ends at 97.57 GiB used (model 80.76 + q8→fp16
acceleration cache 10.58 + driver/arena overhead) and the session adds
~29.4 KiB/token at pc=8192.

Session marginal cost at pc=8192 (measured slope, 21 ratio-4 layers):

| component | KiB/token | note |
|---|---|---|
| attn-compressed KV (fp32) | 10.5 | `DS4_GPU_ATTN_COMP_CACHE_F16` is 0 off-Apple |
| indexer-scores scratch | 8.0 | `comp_cap x pc` fp32 |
| comp_mask scratch | 8.0 | `comp_cap x pc` fp32 — **only col 0 ever used** |
| index-comp cache (fp32) | 2.6 | feeds the score kernel |
| attn-comp r128 (fp32) | 0.3 | |

Two bit-exact fixes in `ds4.c`:

1. **comp_mask `comp_cap x pc` → `comp_cap x 1` float.** The batched prefill
   path runs the indexed attention kernel (comp_selected, no mask); the
   batched decode-mixed path never receives a mask (`use_comp_mask` is only
   set alongside `use_indexed_comp`); only the per-token fallback and decode
   scratch read it, at column 0. Logits **bit-identical** (0/129280 diffs at
   16K and 64K vs the pre-change binary). Saves 4.3 GiB @512K, 6.4 @768K,
   8.6 @1M.
2. **Prefill chunk 8192 → 4096 for ctx > 512K.** Halves the indexer-scores
   buffer and the batch-HC buffers (~5.2 GiB @1M). Costs ~5-6% prefill on
   runs that are already indexer-bound. The 8192 default is unchanged at
   <=512K (64K prefill tps within noise, logits bit-identical). The
   4096-vs-8192 chunk logit difference is **pre-existing**: the raw SWA cache
   holds the whole current ubatch raw, so the chunk size sets the
   uncompressed attention window (verified the old binary produces the
   identical max-delta 3.24 at 16K).

Measured memory (session create):

| ctx | pc | used | free | status |
|---|---|---|---|---|
| 512K | 8192 | ~114.3 GiB | ~7.4 GiB | OK |
| 768K | 8192 (explicit) | 122.25 GiB | 1.75 GiB | OK, tight |
| 768K | 4096 (auto) | 114.98 GiB | 9.02 GiB | OK |
| 1M | 4096 (auto) | 119.34 GiB | 4.66 GiB | OK (was OOM) |

Measured throughput (new binary): 64K frontier prefill 209.9 t/s @768K
session, 205.3 t/s @1M; decode 14.7 t/s @768K, 14.4 t/s @1M (unchanged
from the 13-16 t/s baseline). Full-prompt prefill at 512K+ is indexer-bound
(linear in n_comp): at 384K full-prompt is 124.7 t/s (54 min); 512K ≈
95 t/s (~1.5 h), 768K ≈ 64 t/s (~3.3 h), 1M ≈ 48 t/s (~6 h) estimated from
the per-chunk cost model. CSV: `speed-bench/strix_halo_maxctx.csv`.

The managed-KV decision now also flips later (the estimate shrank):
384K sessions may stay device-resident instead of managed.

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
| `DS4_METAL_MEMORY_REPORT` | (unset) | print used/free GPU memory after model load and after session create |

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
a8a74d7 rocm: register-resident MoE WMMA epilogues + faster small-batch gate (bit-exact)
fd054d2 rocm: fp16 indexer q for prefill scores (bit-exact, ~6% score stage)
ac9c87a rocm: indexer top-k via CUB radix tree (1.3-1.4x, bit-exact)
3f74827 rocm: rewrite indexer scores WMMA128 kernel (1.47x, bit-exact)
bb95f73 docs: add Strix Halo performance investigation
0bccfdd rocm: improve managed-KV threshold, arena limit, and reserve
d897da4 bench: add AMD Strix Halo resident and SSD streaming results
de65cae makefile adjustments for fedora44 build
```

## DSpark Speculative Decoding (2026-08-18, post-upstream-merge)

Merged upstream `main` through `84cc882` (`rocm: enable DSpark speculative
decoding`). DSpark is opt-in via `--mtp <support.gguf> --dspark`; the support
GGUF (~5.6 GiB) is `DeepSeek-V4-Flash-DSpark-support-0731.gguf`
(`./download_model.sh ds4f-dspark`). Flags: `--dspark`,
`--dspark-confidence F` (default 0.7 on ROCm/CUDA), `--dspark-strict`
(target-only decode for reproducibility checks).

Short-context greedy measurement on Strix Halo (ctx 32768) shows DSpark
*slower* than ordinary decode here: code 14.8 vs 17.2 t/s, a one-word
factual prompt 10.4 vs 17.0 t/s. The draft + verification overhead only
amortizes on longer, highly predictable continuations. Run commands in
STRIX_HALO.md §6; full DSpark contract in README.md.

## WMMA two-pass indexed attention (2026-08-11)

`attention_indexed_mixed_heads16_wmma_kernel` replaces the online kernel on
the fast path (`!g_quality_mode`, `n_head<=64`, `top_k<=512`). Block = 1 token
x 16 heads, 256 threads (8 warps), grid (n_tokens, n_head/16), 29.7 KiB
shared (fp16 scores `sw[16][768]` + row arrays; no `__launch_bounds__` -- the
(256,2) target faults in the -g build via register-limit spills on this ROCm,
and the measured occupancy is fine without it). The three passes:

1. **Score GEMM**: `scores[16][n_pad] += q16 x kv^T` with 16x16x16 fp16 WMMA
   (fp32 accumulators). A fragments are built manually from the fp32 q buffer
   (`batch_q_half` is NULL on ROCm); B fragments gather fp16 kv per comp from
   the raw/comp buffers (raw rows via the ring-offset `raw_rows[]`, comp rows
   via the filtered topk). 6 N-tiles per warp -> 768 comps max (n_score =
   top_k 512 + raw 128 + pad). Scores are stored fp16 in shared `sw[16][768]`.
2. **Two-pass softmax** (2 heads per warp, seeded by the sinks): per-lane
   max/sum over the fp16 scores, then `warp_max_f32`/`warp_sum_f32` with a
   **lane-0 broadcast** (`__shfl_sync(FULL_WARP_MASK, v, 0)`) -- the shfl_down
   butterfly leaves the true result only in lane 0 (lanes 16-31 double their
   segment on out-of-range sources), and without the broadcast the weights
   written by lanes 16-31 come from segment sums (the "race" that stalled the
   first integration attempt). Weights are fp16, zero-filled over the full
   768 comps (production n_score is not a multiple of 16).
3. **V GEMM**: `out[16][512] = w x kv` with fp16 weights from shared and
   fp16 kv, fp32 accumulators; the B-fragment gather guards `comp < n_score`.

Numerics: heads match a CPU fp16 two-pass model at 3e-4 (the residual is the
WMMA accumulation order); the scores alone are bit-exact vs the fp16 model.
End-to-end logits are argmax-stable: top-5 identical, top-5 prob mass 0.9932
vs 0.9937, mean |logit delta| 0.32 (the 43-layer accumulation of the
intentional fp16 q/kv/score/weight change).

Performance (prefill t/s): 4112 239.0->242.8, 64K 229.4->233.0, 128K
202.4->205.8 (+1.6-1.7%). The attention stage itself drops ~2.4x (216 -> ~90
ms/layer at the 64K tail), but n_score is capped at 640 by top_k=512 so the
attention is a minority of the per-chunk time. The topk sort is skipped for
this path (the two-pass softmax is order-independent), saving its 2-5
ms/layer-chunk. `g_quality_mode` keeps the bit-exact online kernel.

The design was first validated in a standalone harness (`/tmp/attnwmma.cu`,
`/tmp/attnwmma3.cu`, 2.2-3.5x the online reference at max|d| ~5e-5) before
the production port; the port was debugged against real dumps (env
`DS4_DUMP_HEADS`, since removed) with a CPU fp16 model of the exact kernel
semantics (row construction, visibility, fp16 rounding).

## fp16 compressed-KV cache: validated, rejected for prefill perf (2026-08-11)

The ROCm attention kernels now accept `comp_kv_f16` end-to-end (the WMMA is
templated `<0>/<1>` on it; the online/decode/fallback/static/masked kernels
and the cublas kv-pack convert fp16 rows at their read sites; the engine's
f32->f16 store path and fp16 cache allocation were already complete from the
Metal side). Flipping `DS4_GPU_ATTN_COMP_CACHE_F16` to 1 on ROCm:

- halves the compressed-KV footprint (-5.25 GiB at 1M ctx; 1M at pc=8192
  becomes feasible);
- produces **bit-identical logits** to the fp32-cache WMMA path
  (max|d| = 0.000000 at 4112 -- the f32->f16 store rounds exactly like the
  in-kernel `__float2half`);
- but costs **~6% prefill** at every context (4112 242.7->229, 64K
  233.0->219, 128K 205.8->192): the WMMA's score/V B-fragment gather reads
  one kv element per comp per tile, and each lane's 2-byte fp16 load touches
  the same sectors as the 4-byte fp32 load (the comp rows are strided), so
  halving the bytes buys nothing while the __half conversion and narrower
  loads add ALU. No decode gain either (decode is matmul-bound).

The cache stays F32 by default on ROCm; the fp16 option remains one flag
flip away for memory-budget-critical 1M sessions.
