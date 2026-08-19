# DSpark ROCm Kernel Optimization

Goal: optimize the DSpark speculative-decoding GPU path on ROCm (Strix Halo /
gfx1151). Reference for the feature: README.md "DSpark Speculative Decoding";
the ROCm kernels were enabled upstream in `84cc882`.

## Measured baseline (256-token code task, ctx 32768)

`DS4_DSPARK_STATS=1` on the 256-token code task gives, per run:

| component | ms | note |
|---|---|---|
| target (M=1 one-token decode, 256 tokens) | 13984 | ~55 ms/token |
| **verify (main model, M=2.5 avg batch)** | **11272** | ~98 ms/token — 1.8x M=1 |
| propose (draft model, M=6) | 2256 | 46 draft cycles |
| snapshot | 16 | |
| saved (55 accepted draft tokens) | 3830 | |
| **net** | **-9726** | DSpark loses ~9.7 s / 256 tok |

Draft propose breakdown (`DS4_DSPARK_STAGE_PROFILE=1` + stats), per cycle:

| part | ms/cycle | total ms |
|---|---|---|
| stage chain (3 stages) | 29 | 1338 |
| - of which FFN (routed_moe dominant) | 9 | 414 |
| - of which attn_output_hc | 4.4 | 202 |
| - of which q_path | 2.9 | 133 |
| - of which chain overhead (launch/sync/copies) | ~10 | ~480 |
| base logits (final head, 128K vocab @ M=6) | 10.9 | 500 |
| cache seed (one-time, prefill KV build) | — | 312 |
| markov argmax | 1.2 | 56 |
| confidence probe | 0.6 | 29 |
| setup | 0.5 | 21 |

## Conclusions from the profile

1. The two *named* DSpark kernels are not the bottleneck:
   - `dspark_markov_argmax_kernel` (indexer.cuh): 56 ms total (0.5% of overhead).
   - `attention_noncausal_raw_batch_heads_kernel` (attention.cuh): 0.08 ms/stage.
   They are naive scalar kernels and are being optimized anyway (self-contained,
   low-risk; the draft output is verified downstream so exact rounding is not
   required), but the payoff is small.

2. The real overhead is the **shared main-model kernels running at tiny batch**:
   - **Verification** runs the full 43-layer model at M=5. Measured with
     `--dspark-confidence 0` (fixed 5-token blocks): 271 ms per 5-token verify
     = 54.2 ms/token — exactly the M=1 cost (54.6 ms/token), *not* faster.
     Since a verify costs the same as decoding the tokens normally, DSpark can
     only win if the small-batch (M<=16) path becomes *cheaper* per token than
     M=1 (currently the batch kernels are launch-latency-bound at M=5, no more
     efficient than the per-token WMMA decode path). 11.3 s of the 13.5 s
     overhead, and it is the *only* thing that can make DSpark a win.
   - **Draft FFN** runs the full 256-expert routed MoE at M=6 (`routed_moe`
     is ~5.5 ms of a ~7.5 ms FFN even at M=14).
   - **Draft final head** computes 128K-vocab logits at M=6 (10.9 ms/cycle).
   - **Stage-chain overhead** ~10 ms/cycle: `ds4_gpu_end_commands()` is a full
     `cudaDeviceSynchronize()` on ROCm, and each draft cycle has ~5-6 such
     sync points plus ~60 tiny M=6 kernel launches.

3. `--dspark-confidence` already prunes low-confidence suffixes; with avg
   0.29 accepted tokens/cycle the scheduler skips 146/192 cycles and DSpark is
   still a net loss. Kernel work alone cannot make DSpark win on this workload —
   it reduces the overhead, not the acceptance ceiling.

## Work plan (in value order)

- [x] Profile + document (this file).
- [x] **Markov argmax kernel**: vectorized the rank-256 int8 dot with float4
      state loads. Bit-exact (identical proposals/acceptance), but **neutral**
      (55.62 → 55.27 ms over a 256-token run, −0.6%): the kernel is bound by
      re-reading the 34.8 MB markov W2 matrix from DRAM once per proposed
      token, not by the ALU dot.
- [x] **Small-batch q8 token tile** (the first real win): the q8 GEMM and the
      grouped-q8 attention output hardcoded a 32-token tile, so M=5-6 batches
      staged 5-6x the rows they used. Sized the tile to the batch
      (2/4/8/16/32), bit-exact. Measured on the 256-token DSpark code task
      (16K ctx): prop_logits 553 → 296 ms (−47%), verify_layer 11164 → 9161 ms
      (−18%), net_saved −9726 → −7245 ms, generation 9.29 → 10.21 t/s (+10%).
      64K prefill unchanged (250 t/s). The grouped-q8 attention-output tile
      is neutral on this benchmark (not a bottleneck there).
- [ ] **Non-causal attention kernel**: tile the score computation (shared-memory
      staging, vectorized dot) instead of the per-thread full-K loop.
- [ ] **Stage-chain overhead**: reduce `cudaDeviceSynchronize()` count in the
      propose path (share one command buffer where the CPU does not need the
      result), and check the draft-token upload path for synchronous copies.
- [ ] **Verification M=5 path**: after the small-batch q8 tile fix the verify
      is 9161 ms / 115 tokens = 79.6 ms/token (fixed M=5 ≈ 44 ms/token, now
      *cheaper* than M=1's 54.6 ms/token). Remaining verify cost is the
      routed-MoE hotlist at M=5 (~2 ms/layer), the HC split/sinkhorn, and the
      score/attention -- all launch-latency-bound tiny kernels. Note the
      binding constraint is now the acceptance rate (0.29 tokens/cycle): a
      5-token verify (222 ms) only saves ~79 ms of accepted-token value, so
      kernel work alone cannot make DSpark a win.
- [ ] **Draft final head @ M=6**: check the 128K-vocab output matmul dispatch.
- [ ] **Draft routed MoE @ M=6**: check the small-batch hotlist config.

## Measurement

Each step is validated with the 256-token code task (`/tmp/code_task.txt`):

```sh
./ds4 --rocm [--dspark --mtp ~/.cache/ds4/models/DeepSeek-V4-Flash-DSpark-support-0731.gguf] \
  -m ~/.cache/ds4/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --prompt-file /tmp/code_task.txt --temp 0 -n 256
```

with `DS4_DSPARK_STATS=1` (net_saved_ms) and `DS4_DSPARK_STAGE_PROFILE=1`
(per-stage breakdown). Success = `net_saved_ms` becomes less negative without
changing the generated text (argmax of the *verifier* must stay authoritative;
draft-side rounding is free to change).
