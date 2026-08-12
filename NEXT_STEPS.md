## Context

Working on ds4 (DeepSeek V4 Flash inference engine) on a Strix Halo machine:
- Framework Desktop, AMD Radeon 8060S (gfx1151, RDNA 3.5), 125 GB RAM, 124 GiB GTT
- Fedora 44, Linux 7.1.7, ROCm 7.x
- Repo: `~/src/github.com/spencergilbert/ds4` on branch `fedora44-ds4`
- Build: `make strix-halo -j$(nproc)`
- Model at `~/.cache/ds4/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (80.76 GiB resident)

## What's Done

0. **Indexer scores kernel 3.65× (DONE 2026-08-11)** — the prefill score
   stage (`indexer_scores_wmma128_kernel_t`) was issue/latency-bound at
   ~8.5 TFLOPS: every block staged the fp16 q through `a_sh` shared memory
   and hit 64 per-head `__syncthreads()` barriers. Rewrote it as a
   direct-load kernel (`indexer_scores_wmma128_direct_kernel` in
   `rocm/ds4_rocm_indexer.cuh`):
   - a-fragments load straight from the global fp16 q (row stride
     `n_head*head_dim`) — no a_sh staging, no barriers
   - two heads interleaved per iteration for MMA ILP (4 accumulator pairs
     in flight; 4-head spills registers and is 10x slower)
   - weights preloaded to shared once; `b_sh` (fp16 index_comp) staged as
     before; causal mask and fully-masked early-out unchanged
   - **q16 head-major transpose**: the QAT step now writes the fp16 q as
     `[head][token][dim]`, so the direct kernel's 16x16 a-tiles are
     512-byte contiguous loads instead of 16 scattered rows (bit-exact;
     score stage 60.8→53.8 ms at the 64K tail, 421→391 ms at 98K×8192
     micro-bench = 33.7 TFLOPS)
   Micro-bench (`/tmp/idxbench3/6/7/8.cu`): 1538 → 391 ms at 98304 comps ×
   8192 tokens (8.6 → 33.7 TFLOPS; MMA-only ceiling ~43). **Bit-identical**
   logits at 16K (incl. a 16-token partial-tile tail chunk), 64K (0/129280
   diffs vs the pre-change binary). The direct kernel is used for all
   n_tokens > 1 (partial tiles read in-bounds of the pc-sized q buffer;
   out-of-range rows compute garbage that the token guards drop; weights
   loads are clamped).
   End-to-end prefill (8192-token chunks): 8K 251.2→252.7, 32K 237.3→243.0,
   64K 219.3→229.4 (+4.6%), 128K 185.5→202.4 (+9.1%), 384K 124.7→142.7
   (+14.5%, 54→46 min; the q16 transpose adds ~1% more at 384K). Decode
   unchanged (~13 t/s). Production stage
   profile at 64K tail (comp=16384, 8192-token chunk): score 181→54
   ms/layer; top-k 49 ms, indexed attention 216 ms are now the bigger
   indexer costs (both probed in `/tmp/attnbench*`/`/tmp/topkbench*`:
   attention is reduction-bound at its online-softmax floor — the 5-shuffle
   warp reduction per row; the top-k 32-bit-key radix
   idea is blocked by score+index needing ~50 bits). CSV/SVG:
   `speed-bench/strix_halo_idx_direct.csv` (online) and
   `speed-bench/strix_halo_wmma.csv` (WMMA two-pass, item 0.5).

0.5 **Indexed attention WMMA two-pass rewrite — DONE 2026-08-11.**
   `attention_indexed_mixed_heads16_wmma_kernel` replaces the online kernel on
   the fast path (`!g_quality_mode`, `n_head<=64`, `top_k<=512`): block = 1
   token × 16 heads, 256 threads (8 warps), grid (n_tokens, n_head/16).
   Score GEMM C[16][n_pad] = fp16 q/kv (A built manually from the fp32 q
   buffer — `batch_q_half` is NULL on ROCm) with fp32 accumulators → two-pass
   softmax (seeded by sinks, warp-max + broadcast) → V GEMM from shared fp16
   weights. fp16 scores/weights are the first intentional logit change;
   heads match a CPU fp16 model at 3e-4, end-to-end logits are argmax-stable
   (top-5 identical, top-5 prob mass 0.9932 vs 0.9937; mean logit delta
   0.32 is the 43-layer accumulation of the fp16 change). The topk sort is
   skipped (order-independent two-pass softmax).
   - **performance**: 4112 239.0→242.8, 64K 229.4→233.0, 128K 202.4→205.8
     (+1.6-1.7%); the attention stage itself drops ~2.4× (216→~90 ms/layer
     at the 64K tail) but n_score is capped at 640 (top_k 512 + 128 raw) so
     the attention is already a minority of the per-chunk time.
   - the last-session "race" in the weights was a **missing lane-0 broadcast**
     after `warp_sum_f32`/`warp_max_f32`: the shfl_down butterfly leaves the
     true result only in lane 0 (lanes 16-31 double their segment on
     out-of-range sources) — the online kernels broadcast
     `__shfl_sync(FULL_WARP_MASK, v, 0)` after each reduce; the WMMA kernel
     originally did not, so lanes 16-31 wrote weights from segment sums.
   - other production deltas vs the micro-bench: full-range weight
     zero-fill (padded comps), V B-fragment `comp<n_score` guard, no
     `__launch_bounds__` (the (256,2) target faults in the -g build —
     register-limit spills on this ROCm), the score
     scratch stays per-block shared (fp16, 24 KiB) — no global scratch.
0.6 **Top-k chunk size adaptive (DONE 2026-08-12)** — the CUB radix tree
   used 8192-row chunks everywhere; the 4096-row tree is ~11% faster up to
   16K comps (measured 39.8 vs 45.1 ms at 16384 comps x 8192 tokens in
   `/tmp/topkbench2.cu`, bit-exact 0 mismatches vs the 8192 tree). The tree
   path is now a templated helper dispatched on n_comp (4096 <= 16K comps,
   8192 above); end-to-end +0.2-0.4% at 64K (the gain is limited to the
   tail chunks whose n_comp is 8K-16K). Also: the prefill stage trace stubs
   (`glm_graph_*_prefill_trace_*`) were hardcoded off — now env-gated
   (`DS4_TRACE_INDEXED_PREFILL`, `DS4_TRACE_FULL_PREFILL` + `_ALL`/`_SLOW`).
0.7 **Indexed-attention DRAM bound — characterized (2026-08-12).** The
   per-k-tile shared staging variant was also tested and is definitively
   worse: 4112 218 vs 242.5 t/s and the 64K-tail attention 580-598 vs
   307 ms (bit-identical logits). The __syncthreads between the staging and
   the MMAs serialize the scattered DRAM loads, destroying the cross-warp
   latency hiding the no-staging kernel gets for free; a double-buffered
   staging cannot fit the 64 KiB LDS (2x24 KB kv + 24 KB sw). The
   no-staging WMMA is the final design; the attention is DRAM-bound and
   stays at its floor.
0.8 **Score-kernel occupancy — register-blocked too (2026-08-12).** The
   indexer-scores direct kernel is 1 block/CU (42 KB shared + 173 VGPRs;
   2 blocks need <=32 KB + <=128 regs). __launch_bounds__(256,2) does not
   help (the compiler keeps 231 VGPRs -- still 1 block/CU; the 64K-tail
   attention stage unchanged at ~312 ms). The one-head variant (fewer live
   WMMA fragments to fit 128 regs) would lose the two-heads-interleaved
   ILP that made the direct kernel 3.65x faster -- the same register-vs-
   ILP wall as the attention. The fp16 index_comp was also quantified:
   the comps are re-read ~2 GB/chunk but that is ~8 ms of the 54 ms score
   stage (the q16 reads are already L2-shared and the kernel is
   MMA-latency-bound); the fp16 saves ~0.5% total for the ~15-site
   plumbing. The indexer (score 54 + topk 40 + attention ~205 ms/layer at
   the 64K tail) is at its floor; the dense matmuls (q_path 80 + output_proj
   105 ms/layer) are now the dominant remaining cost.
0.9 **Dense GEMMs: Q8_0 batch 4w->8w landed (2026-08-12, `73f0a6a`); the
   output_proj's low-rank stays on cublas.** The batch Q8_0 GEMM (q_path
   etc.) went from ~20% to ~24% of the fp16 WMMA peak via the 8-warp tile
   (+13-18%, q_path 80->72 ms/layer, bit-exact). The output_proj (104
   ms/layer = ~61 ms low-rank + ~53 ms dense) uses the cublas strided-
   batched low-rank GEMM at the production shape (8 groups x 4096 dims x
   1024 rank, ~9 TFLOPS). A WMMA grouped prototype (heads staged in
   shared, Q8_0 dequant): the 512-dim group variant hits 12.9 TFLOPS, but
   the production's 4096-dim groups need a chunked K-loop -> 7.1 TFLOPS
   (the per-chunk syncs; a double-buffered variant drops to 5.4 with 1
   block/CU). The cublas stays for the output_proj. (The grouped scalar
   fallback kernel is ~2 TFLOPS.)
0.10 **Indexed-attention DRAM bound — characterized (2026-08-12).** The
   profile (`DS4_ROCM_LAYER_STAGE_PROFILE=1`) shows the 64K-tail attention
   stage (score+topk+WMMA) at 282-312 ms/layer (vs 78 ms at 4112): the WMMA
   is DRAM-bandwidth-bound on the scattered topk comp-row gathers -- the
   4 head-blocks per token pull the same ~640 selected rows from a cache
   that grows to 33 MB/layer (the L2 is 2 MB), ~42 GB of scattered traffic
   per layer-chunk at ~50% DRAM efficiency. Tried and rejected: grid-order
   swap for L2 sharing (no effect -- the L2 is too small for the wave),
   k-loop unroll 2 (no effect), 4-head-group fusion with per-k-tile shared
   staging (cacc[4][6] = 192 VGPRs spills), 2-group fusion (also spills).
   The register file (256 VGPRs/thread) blocks the fusion; the attention
   stays at its current floor. A lower-register WMMA (e.g. staging the kv
   in shared per k-tile in the existing 16-head block) remains the open
   idea but the micro-bench showed the staged variant was 4.6x slower.
   Occupancy work (2 blocks/CU) also ruled out: MFMA-latency-bound.
0.8 **WMMA occupancy (2 blocks/CU) — ruled out (2026-08-12).** The kernel
   uses 158 VGPRs -> 1 block/CU (2 blocks need <=128); __launch_bounds__(256,2)
   spills and faults on this ROCm. Restructured the score GEMM to 4 N-tiles
   per warp over two 512-comp passes + the V to 2 D-tiles over two 256-dim
   passes -> 125 VGPRs, 2 blocks/CU, bit-identical logits -- but SLOWER
   (4112 240.2 vs 242.5, 64K 229.3 vs 234.0): the kernel is MFMA-latency-
   bound at its 6-tile ILP, not occupancy-bound; halving the per-warp MMA
   ILP costs more than the doubled occupancy gains. Reverted. The WMMA
   attention is near its floor; bigger levers are the cross-layer stage
   overlap (score+topk of layer L+1 overlapping layer L's attention) or the
   score stage's fp16 index_comp.
0.8 **fp16 compressed-KV cache — plumbing landed, cache stays F32
   (2026-08-11).** All ROCm attention kernels now accept `comp_kv_f16`
   (the WMMA is templated `<0>/<1>`; the online/decode/fallback/static/
   masked kernels and the cublas kv-pack read fp16 at their sites). The
   engine's f32->f16 store + fp16 cache allocation were already complete
   (Metal). Flipping `DS4_GPU_ATTN_COMP_CACHE_F16` to 1 halves the
   compressed KV (-5.25 GiB @1M, unblocks 1M@pc=8192) with logits
   bit-identical to the fp32-cache WMMA path — but costs ~6% prefill at
   every context (4112 242.7->229, 64K 233.0->219, 128K 205.8->192): the
   WMMA's per-comp B-fragment gather is sector-bound, not byte-bound, so
   2-byte loads buy nothing. No decode gain (decode is matmul-bound). The
   cache stays F32 by default; the flag remains as a documented option for
   memory-budget-critical 1M sessions. See `docs/strix-halo-perf.md`
   §"fp16 compressed-KV cache".

1. **Max usable context 512K → 1M (DONE 2026-08-11, `b210a95`)** — the model's full
   `context_length` (1M) now creates a session and prefills on Strix Halo. The
   previous ceiling was ~512K (768K/1M OOM'd at session create). Two bit-exact
   memory-budget changes in `ds4.c` (measurements via
   `DS4_METAL_MEMORY_REPORT=1`, added to the session-create path):
   - **comp_mask scratch shrink**: the `comp_cap x pc` fp32 mask buffer is only
     consumed per-token — the batched prefill path runs the indexed attention
     kernel (which takes `comp_selected`, not the mask), the batched
     decode-mixed path never receives a mask (`use_comp_mask` is only set
     alongside `use_indexed_comp`), and the per-token fallback / decode scratch
     read column 0 only. Sized it `comp_cap x 1` float. Saves 4.3 GiB @512K,
     6.4 GiB @768K, 8.6 GiB @1M. **Bit-identical** logits (0/129280 diffs at
     16K and 64K vs the pre-change binary; the launch-side `n_tokens x n_comp`
     byte checks still gate any future batch-mask use).
   - **Prefill chunk 8192 → 4096 for ctx > 512K**: the per-chunk indexer-scores
     buffer (`comp_cap x pc`) halves (another ~4.3 GiB @1M), and the batch-HC
     buffers shrink too. Costs ~5-6% prefill on runs that are already
     indexer-bound at these lengths; the 8192 default is unchanged at <=512K
     (no regression: 64K prefill tps within noise, logits bit-identical). Note
     the 4096-vs-8192 chunk logit difference is **pre-existing** engine
     behavior (the raw SWA cache holds the whole current ubatch raw, so the
     chunk size sets the uncompressed attention window; verified the old binary
     produces the identical max-delta 3.24 at 16K).
   Measured (new binary): 768K session used=122.25 GiB / free=1.75 GiB
   (8192 chunks, explicit) or 114.98 / 9.02 (auto 4096); 1M used=119.34 /
   free=4.66 (auto 4096). Prefill 64K frontier at 768K/1M sessions:
   209.9/205.3 t/s; decode 14.7/14.4 t/s. CSV: `speed-bench/strix_halo_maxctx.csv`.
   **Update (2026-08-12, `78335e2`)**: the <=256K ROCm default is now the
   16384-token chunk (+1.5-1.6%: 64K 239.5 vs 235.7, 128K 208.9 vs 205.8;
   raw ring grows to match -- `metal_graph_raw_cap_for_context`/planner twin
   scale past 8192 only when the ubatch exceeds it). The 16K chunk OOMs at
   384K (per-chunk scratch ~4-5 GiB over 8K), so 256K-512K keeps 8192 and
   >=512K keeps 4096. 4112 logits bit-identical (single chunk); 64K
   frontier argmax-stable (max|d|=2.40 vs the 8K chunk -- the documented
   chunk-boundary logit behavior).

2. **MoE prefill kernels (DONE 2026-08-11, `a8a74d7`)** — the routed MoE is the
   biggest per-layer cost (~43-49%); for agentic turn prefill (small token
   batches) the scalar cold-expert path dominated at ~1.8-2.6 TFLOPS. Three
   bit-exact changes in `rocm/ds4_rocm_moe.cuh`/`_launch.cuh`:
   - **Gate/up IQ2 WMMA hotlist**: register-resident epilogue (drop the 16 KB
     fp32 C-tile shared round trip; per-wave fragment writes via the measured
     gfx1151 accumulator layout). Shared 20→4 KB, kernel 166→133 ms/layer.
   - **Down Q2K WMMA hotlist**: same epilogue; 90→76 ms/layer.
   - **Scalar cold-expert gate (tile8 rowspan)**: drop the 37 KB sxq shared
     staging (read xq via L2) and row span 1024→256. 200-token gate
     22.7→15.3 ms/layer (-33%).
   Logits **bit-identical** at 512/2048/65536 ctx (0/129280 diffs). Prefill
   t/s: 128 41.7→42.9, 512 125.7→131.7, 2K 195.8→206.5, 8K 236.5→251.3,
   32K 224.3→237.3, 64K 205.3→218.5, 128K 179.9→185.5, 384K 121.5→124.7.
   Decode unchanged (13.0 t/s). CSV/SVG: `speed-bench/strix_halo_moe_epi.csv`.

3. **Benchmarks produced** at 64K ctx: `speed-bench/strix_halo.csv` (resident sweep
   2K→64K, 157–212 t/s prefill, 13–16 t/s gen) and `speed-bench/strix_halo_ssd.csv`
   (SSD streaming sweep, 50–59 t/s prefill, 10–12 t/s gen). Both have SVGs.

4. **Code fixes committed** (`0bccfdd`):
   - Managed KV threshold: removed hard 8 GiB cutoff in `rocm/ds4_rocm_runtime.cuh`
     and `ds4_cuda.cu` — now consults free device memory instead
   - Arena span limit: `total/3` → `total*3/4` in `rocm/ds4_rocm_runtime.cuh`
     for SSD streaming headroom
   - Managed KV reserve: `total/4` → `total/8` so more contexts keep device-resident KV

5. **Extensive testing** of SSD streaming at 384K–1M ctx with various cache sizes
   (see `docs/strix-halo-perf.md`). Ceiling is ~56 GB expert cache at 512K ctx,
   ~67 GB at 512K. Auto budget (69 GiB) OOMs the arena. All streaming at >64K
   thrashes (26–415× read amplification from q8-fp16 cache churn). **SSD streaming
   is now unnecessary for DeepSeek up to 1M** — the resident session fits.

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

| ctx | original | final (score+topk+fp16q) | Δ |
|---|---|---|---|
| 64K | 192.6 t/s | 198.3 t/s | +3.0% |
| 128K | 163.2 t/s | 172.2 t/s | +5.5% |
| 384K | 102.0 t/s | 114.9 t/s | +12.6% (64 → 57 min) |

Sparse sweep CSV/SVG: `speed-bench/strix_halo_idx.csv` + `_ts.svg` (2K, 32K,
64K, 128K, 384K). All three changes are bit-identical (logit JSON diff at
8K/64K).

### 4. Prefill chunk 8192 for ROCm (DONE 2026-08-10)

`--prefill-chunk 8192` (and now the ROCm default in `ds4_prefill_cap_for_prompt`)
measures **+5-6%** everywhere (better GEMM utilization, fewer launches):
64K 198.3 → 208.2, 128K 172.2 → 179.9, 384K 114.9 → 121.5 t/s. 16384-token
chunks exceed the raw SWA cache cap (8192) and fail. Raw SWA cache grows to
8192 rows (+~1 GiB across layers).

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

### 2. fp16 indexer q (DONE 2026-08-10)

The QAT step (`indexer_hadamard_fp4_kernel`) now also emits an fp16 copy of
the rounded indexer q into a new `batch_indexer_q_half` graph buffer; the
prefill scores kernels (`indexer_scores_wmma128_kernel_t<__half>`) read it
directly (skipping in-kernel `__float2half`, halving a_sh staging traffic).
The fp32 q is still written for the per-token fallback / decode / debug.

**Bit-identical** logits at 8K. Score stage 120 → 113 ms at 64K,
253 → 238 ms at 128K tail (~6%). The index_comp (k) side was NOT converted:
it would require changing the compressor output + cache format + ~15
consumer sites (incl. snapshot restore) for ~2-4% more — not worth it.

### 3. Bigger tiles / occupancy (NOT pursued)

64-token blocks exceed the 64 KiB LDS budget at the 136-stride (69.6 KiB).
The score kernel is ~8.5 TFLOPS (vs ~15 fp32 peak) — likely issue/latency
bound rather than bandwidth bound; the fp16-q change captured the cheap
memory-side gain.

### 4. Validate & benchmark

- Logit dump comparison vs the GEMM path at 2K–64K (tolerance 1e-3) — the
  kernel rewrite is bit-exact so far.
- Sweep 2K→393K with `--gen-tokens 0`, produce
  `speed-bench/strix_halo_fa.csv` + SVG (rename: the win is the indexer
  kernel, not FA).
- 384K confirmation run.

## Max-Context Memory Budget (2026-08-11)

Measured with `DS4_METAL_MEMORY_REPORT=1` (now also printed at session
create). The session's per-token marginal cost is ~29.4 KiB/token at
pc=8192, dominated by: attn-compressed KV fp32 (10.5), indexer-scores
scratch (8), comp_mask scratch (8), index-comp fp32 (2.6). The old
`comp_cap x pc` comp_mask was **dead weight** (only column 0 is ever
consumed). After the two fixes the marginal cost is ~17.4 KiB/token:

| ctx | pc | used after session | free | status |
|---|---|---|---|---|
| 512K | 8192 | ~114.3 GiB | ~7.4 GiB | OK (was 3.1) |
| 768K | 8192 (explicit) | 122.25 GiB | 1.75 GiB | OK, tight |
| 768K | 4096 (auto) | 114.98 GiB | 9.02 GiB | OK |
| 1M | 4096 (auto) | 119.34 GiB | 4.66 GiB | OK (was OOM) |
| 1M | 8192 | — | — | OOM (short ~0.6 GiB) |

Remaining levers if headroom is ever needed again (all would change logits
slightly except the first):
- **fp16 compressed KV** (Metal already stores the attn-comp cache fp16;
  the ROCm attention kernels currently reject `comp_kv_f16`, so this needs
  fp16-read support in the 4 attention kernels): −5.25 GiB @1M.
- **fp16 index_comp cache**: −1.3 GiB @1M (score inputs change).
- **fp16 indexer-scores buffer**: −4.1 GiB @1M (top-k rank can flip near
  the boundary).
- **q8→fp16 cache partial yield** at session create: up to −10.6 GiB but
  prefill drops toward 148 t/s (measured −41% with the cache disabled).

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

# Memory-budget diagnosis (used / free after model load and after session):
DS4_METAL_MEMORY_REPORT=1 ./ds4-bench -m <model> --prompt-file /tmp/promessi_x6.txt \
  --ctx-start 2048 --ctx-max 2048 --ctx-alloc 1048576 --gen-tokens 0

# Max-context smoke (1M session, 2 frontiers, ~1 min after model load):
./ds4-bench -m <model> --prompt-file /tmp/promessi_x6.txt \
  --ctx-start 2048 --ctx-max 2048 --ctx-alloc 1048576 --gen-tokens 0

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
