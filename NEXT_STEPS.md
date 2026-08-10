## Context

Working on ds4 (DeepSeek V4 Flash inference engine) on a Strix Halo machine:
- Framework Desktop, AMD Radeon 8060S (gfx1151, RDNA 3.5), 125 GB RAM, 124 GiB GTT
- Fedora 44, Linux 7.1.7, ROCm 7.x
- Repo: `~/src/github.com/spencergilbert/ds4` on branch `fedora44-ds4`
- Build: `make strix-halo -j$(nproc)`
- Model at `~/.cache/ds4/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (80.76 GiB resident)

## What's Done

1. **Benchmarks produced** at 64K ctx: `speed-bench/strix_halo.csv` (resident sweep
   2K→64K, 157–212 t/s prefill, 13–16 t/s gen) and `speed-bench/strix_halo_ssd.csv`
   (SSD streaming sweep, 50–59 t/s prefill, 10–12 t/s gen). Both have SVGs.

2. **Code fixes committed** (`0bccfdd`):
   - Managed KV threshold: removed hard 8 GiB cutoff in `rocm/ds4_rocm_runtime.cuh`
     and `ds4_cuda.cu` — now consults free device memory instead
   - Arena span limit: `total/3` → `total*3/4` in `rocm/ds4_rocm_runtime.cuh`
     for SSD streaming headroom
   - Managed KV reserve: `total/4` → `total/8` so more contexts keep device-resident KV

3. **Extensive testing** of SSD streaming at 384K–1M ctx with various cache sizes
   (see `docs/strix-halo-perf.md`). Ceiling is ~56 GB expert cache at 512K ctx,
   ~67 GB at 512K. Auto budget (69 GiB) OOMs the arena. All streaming at >64K
   thrashes (26–415× read amplification from q8-fp16 cache churn).

## The 64K Context Cliff Is NOT Reproducible (as described)

The earlier "Key Finding" blamed prefill attention: "At compressed-KV rows above
~16K, rocBLAS picks a suboptimal tiling strategy and throughput collapses ~72×".
**This does not reproduce on the current binary.** Measured 2026-08-10:

| ctx | compressed rows | prefill t/s (full prompt, gen 0) | 2048-token chunk at frontier |
|---|---|---|---|
| 64K | 16,386 | 192.5 | ~13 s |
| 66K | 16,642 | — | ~13 s (no cliff at the 65,536 RoPE boundary) |
| 128K | 32,770 | 163.2 | — |
| 200K | 49,154 | 142.7 | — |
| 384K | 98,306 | 102.0 | ~31 s |

Why: DeepSeek V4 Flash has a **compressed-KV indexer**. For `n_comp > 512`
(`DS4_N_INDEXER_TOP_K`), prefill attention uses the **indexed online kernel**
(`attention_indexed_mixed_heads8_online_kernel`) with fixed top-k=512 + the
128-token raw window — its cost is **flat in context** (~108 ms per
layer/chunk at any ctx). The `cublasGemmStridedBatchedEx` S-matrix path
(`attention_prefill_mixed_cublas_tiled`) only runs for `n_comp <= 512`.

The actual context-scaling cost (measured with `DS4_ROCM_LAYER_STAGE_PROFILE` /
`DS4_ROCM_INDEXER_STAGE_PROFILE`, layer 0, 4096-token chunks):

| stage | 64K (comp=16,384) | scales with |
|---|---|---|
| indexer scores (WMMA128 kernel) | ~181 ms/layer | n_comp (linear) |
| indexer top-k | ~26 ms/layer | n_comp |
| indexed attention (top-k 512) | ~108 ms/layer | none (flat) |
| routed MoE, Q path, output proj, ... | ~200–360 ms/layer | none |

So >64K prefill degrades linearly (indexer scores dominate), not catastrophically.

## Done This Session: Indexer Scores Kernel 1.47×

`rocm/ds4_rocm_indexer.cuh::indexer_scores_wmma128_kernel` was the bottleneck
(6.0 TFLOPS effective). Rewrote it:

- 16 tokens → **32 tokens per block** (halves the per-layer q re-reads and
  block count)
- Removed the per-head `c_sh` shared-memory score round trip — the two 16×16
  accumulator fragments now stay in registers through the weighted-ReLU head
  reduction, using the measured rocwmma gfx1151 accumulator layout
  (`row = 2*i + (lane>>4)`, `col = lane & 15` — differs from nvcuda!)
- Double-buffered `a_sh` so the next head's q fetch + fp16 convert overlaps
  the current head's MMA (1 barrier per head instead of 3)
- Launch grid: `(n_comp+127)/128 x (n_tokens+31)/32`

Validated: logits **bit-identical** to the old kernel at 8K and 64K ctx.
Micro-benchmark (`/tmp/idxbench.cu`): 182.6 ms → 129.3 ms (8.5 vs 6.0 TFLOPS)
at 16384 comps × 4096 tokens. Same double-buffer applied to the CUDA
`ds4_cuda.cu` copy (nvcuda layout; not compiled/tested on this machine).

Measured end-to-end (resident, gen-tokens 0):

| ctx | before | after | Δ |
|---|---|---|---|
| 64K | 192.6 t/s | 198.3 t/s | +2.9% |
| 128K | 163.2 t/s | 171.3 t/s | +4.9% |
| 384K | 102.0 t/s | 112.6 t/s | +10.3% (58 min vs 64 min) |

Sparse sweep CSV/SVG: `speed-bench/strix_halo_idx.csv` + `_ts.svg` (2K, 32K,
64K, 128K, 384K).

## rocBLAS / hipBLAS Tuning: Tested, No Effect

- `ROCBLAS_GEMM_ALGO_ALWAYS=1`: 191.3 vs 192.6 t/s baseline — noise.
- `ROCBLAS_TENSORLIB_PATH` / `HIPBLASLT_TENSORLIB_PATH`: gfx1151 libraries exist
  at `/usr/lib64/rocblas/library/` and `/usr/lib64/hipblaslt/library/` (no YAML
  autotune files).
- The hipBLASLt plan machinery (`rocm/ds4_rocm_hipblaslt.cuh`,
  `MAX_WORKSPACE_BYTES`, 8 heuristic candidates) is **dead code** on this branch:
  `hipblaslt_gemm_tn_f16_out_f16` has no callers. fp16 GEMMs go through
  `cublasGemmEx(..., CUBLAS_GEMM_DEFAULT)` → rocBLAS directly.
- The proposed "replace `CUBLAS_GEMM_DEFAULT` when compressed rows > 16K"
  targets the cublas S-matrix attention path, which the indexer model never
  uses beyond n_comp=512.

## Next Steps (in priority order)

### 1. Indexer top-k kernel (DONE 2026-08-10)

Replaced the 4096-row bitonic tree (144-pass full sorts to extract top-512)
with a CUB radix-sort tree over 8192-row chunks:
`indexer_topk_chunk_cub_kernel<8192>` + `indexer_topk_tree_merge_cub_kernel`
+ `indexer_topk_final_merge_cub_kernel` in `rocm/ds4_rocm_indexer.cuh`.
The bitonic 4096 tree stays as a fallback for devices without 64 KiB opt-in
shared memory.

Micro-benchmark (4096 tokens): 26.6 → 20.0 ms at 16K comps (cub4096),
160.4 → 115.4 ms at 98K comps (cub8192, 1.39x), bit-exact vs the bitonic
reference at every n_comp. Production stage profile: 26.6 → 24.8 ms at 64K
(comp=16384), 44 ms at 128K tail (comp=32768). End-to-end 64K 198.9,
128K 171.8 t/s (no regression).

### 2. fp16 index inputs

The scores kernel converts fp32→fp16 in-kernel. A pre-pass that stores q and
`index_comp` as fp16 would halve the dominant memory traffic (~16 GB/layer of
q re-reads at 64K) with **bit-identical** MMA inputs. q is [tokens×64×128]
fp32; index_comp cache is [n_comp×128] fp32.

### 3. Bigger tiles / occupancy

64-token blocks would halve q traffic again (LDS is the constraint: 64KB
limit, ~69.6KB needed at 136-stride; 128-stride is exactly 64KB and risky).
Also try 512-thread blocks.

### 4. Validate & benchmark

- Logit dump comparison vs the GEMM path at 2K–64K (tolerance 1e-3) — the
  kernel rewrite is bit-exact so far.
- Sweep 2K→393K with `--gen-tokens 0`, produce
  `speed-bench/strix_halo_fa.csv` + SVG (rename: the win is the indexer
  kernel, not FA).
- 384K confirmation run.

### Not needed

- FlashAttention HIP kernel (Priority 2 of the old plan): attention is
  already flat via the indexed online kernel; there is no attention cliff.
- hipBLASLt MAX_WORKSPACE / 8-heuristic work: dead code path.

### To test any change

```sh
cd ~/src/github.com/spencergilbert/ds4
make strix-halo -j$(nproc)
rm -f /tmp/ds4.lock  # if stale

# Smoke test (prefill-only, 2 frontiers, ~30s):
./ds4-bench \
  -m ~/.cache/ds4/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --prompt-file /tmp/promessi_x6.txt \
  --ctx-start 2048 --ctx-max 65536 --gen-tokens 0

# Per-stage profiling (indexer stages, layer 0):
DS4_ROCM_INDEXER_STAGE_PROFILE=1 DS4_ROCM_LAYER_STAGE_PROFILE=1 \
DS4_ROCM_LAYER_STAGE_PROFILE_LAYER=0 ./ds4-bench -m <model> \
  --prompt-file /tmp/promessi_x6.txt --ctx-start 65536 --ctx-max 65536 --gen-tokens 0

# Do NOT run two ds4 processes concurrently — single instance lock.
# Kill with: pkill -9 -f ds4-bench; rm -f /tmp/ds4.lock
```

### Success Criteria

- 384K prefill t/s improves over the 102.0 baseline (was claimed 72×
  degraded; it is ~1.9× vs 64K and the kernel rewrite targets the remaining
  gap).
- No logit drift > 1e-3 vs. the old path at 64K (bit-identical so far).
- Output CSV: `speed-bench/strix_halo_fa.csv` + SVG (sweep 2K→393K).

## Reference

- Setup guide: `STRIX_HALO.md`
- Full investigation: `docs/strix-halo-perf.md`
- Existing benchmarks: `speed-bench/strix_halo*.csv`
- ROCm runtime: `rocm/ds4_rocm_runtime.cuh`
- hipBLASLt plans: `rocm/ds4_rocm_hipblaslt.cuh` (dead code on this branch)
- ROCm indexer kernels: `rocm/ds4_rocm_indexer.cuh`
- Model directory: `~/.cache/ds4/models/` (3 GGUF files, symlink `ds4flash.gguf`
  → GLM model — use DeepSeek path explicitly as above)
