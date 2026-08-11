__global__ static void indexer_hadamard_fp4_kernel(float *x, __half *x16, uint32_t n_rows, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    const float out = dsv4_e2m1fn_dequant_dev(fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
    xr[tid] = out;
    if (x16) x16[(uint64_t)row * head_dim + tid] = __float2half(out);
}

__global__ static void indexer_scores_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
    uint32_t c = blockIdx.x;
    uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tokens) return;
    if (causal) {
        uint32_t n_visible = (pos0 + t + 1u) / ratio;
        if (c >= n_visible) {
            if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    float total = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
        const float *kh = index_comp + (uint64_t)c * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) dot += qh[d] * kh[d];
        __shared__ float partial[256];
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        total += fmaxf(partial[0], 0.0f) * weights[(uint64_t)t * n_head + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = total * scale;
}

__global__ static void indexer_score_one_direct_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal) {
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || tid >= 128u) return;
    if (causal) {
        const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_comp;
        if (c >= visible) {
            if (tid == 0) scores[c] = -INFINITY;
            return;
        }
    }

    __shared__ float krow[128];
    __shared__ float partial[4];
    if (tid < 128u) krow[tid] = index_comp[(uint64_t)c * 128u + tid];
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);
        if (lane == 0) partial[warp] = fmaxf(dot, 0.0f) * weights[h] * scale;
        __syncthreads();
        if (tid == 0) total += partial[0] + partial[1] + partial[2] + partial[3];
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}

__device__ __forceinline__ static __half indexer_q_load(const float *q, uint64_t off) {
    return __float2half(q[off]);
}
__device__ __forceinline__ static __half indexer_q_load(const __half *q, uint64_t off) {
    return q[off];
}

template <typename QT>
__global__ static void indexer_scores_wmma128_staged_kernel_t(
        float *scores,
        const QT *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700 || defined(__HIP_DEVICE_COMPILE__)
#ifdef __HIP_PLATFORM_AMD__
    namespace wmma = rocwmma;
#else
    namespace wmma = nvcuda::wmma;
#endif
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 32u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 32u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 32u * 128u; i += 256u) {
                const uint32_t r = i >> 7u;
                const uint32_t c = i & 127u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    /* 32 tokens x 128 comps per block.  a_sh is double-buffered so the next
     * head's global q fetch + fp16 convert overlaps the current head's MMA
     * (one barrier per head), and the two 16x16 accumulator fragments per
     * warp stay in registers through the weighted-ReLU head reduction (no
     * shared-memory score round trip).  The rocwmma 16x16x16 f32 accumulator
     * layout on gfx1151 is: element i of lane l holds (row = 2*i + (l>>4),
     * col = l & 15).  Measured 1.4x faster than the 16-token c_sh round-trip
     * form at 16384 comps x 4096 tokens (8.5 vs 6.0 TFLOPS). */
    __shared__ __half a_sh[2][32 * 136];
    __shared__ __half b_sh[128 * 136];

    const uint32_t lane = tid & 31u;

    float acc0[8], acc1[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) {
        acc0[i] = 0.0f;
        acc1[i] = 0.0f;
    }

    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 136u] = __float2half(v);
    }
    /* Stage head 0 into a_sh[0] before the loop. */
    for (uint32_t i = tid; i < 32u * 128u; i += 256u) {
        const uint32_t r = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t token = tile_t + r;
        __half v = __float2half(0.0f);
        if (token < n_tokens) {
            v = indexer_q_load(q, ((uint64_t)token * n_head + 0u) * head_dim + d);
        }
        a_sh[0][r * 136u + d] = v;
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        /* Prefetch head h+1 into the other a buffer while head h computes. */
        if (h + 1u < n_head) {
            for (uint32_t i = tid; i < 32u * 128u; i += 256u) {
                const uint32_t r = i >> 7u;
                const uint32_t d = i & 127u;
                const uint32_t token = tile_t + r;
                __half v = __float2half(0.0f);
                if (token < n_tokens) {
                    v = indexer_q_load(q, ((uint64_t)token * n_head + h + 1u) * head_dim + d);
                }
                a_sh[(h + 1u) & 1u][r * 136u + d] = v;
            }
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
        wmma::fill_fragment(c0, 0.0f);
        wmma::fill_fragment(c1, 0.0f);
        const __half *a_cur = a_sh[h & 1u];
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a0, a_cur + k0, 136);
            wmma::load_matrix_sync(a1, a_cur + 16u * 136u + k0, 136);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 136u + k0, 136);
            wmma::mma_sync(c0, a0, b_frag, c0);
            wmma::mma_sync(c1, a1, b_frag, c1);
        }

        /* rocwmma 16x16x16 f32 accumulator: element i of lane l holds
         * (row = 2*i + (l>>4), col = l & 15) of the 16x16 tile, so the
         * 8 elements of each fragment cover 8 distinct token rows. */
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint32_t row = 2u * (uint32_t)i + (lane >> 4u);
            const float w0 = (tile_t + row < n_tokens)
                ? weights[((uint64_t)(tile_t + row)) * n_head + h] : 0.0f;
            const float w1 = (tile_t + 16u + row < n_tokens)
                ? weights[((uint64_t)(tile_t + 16u + row)) * n_head + h] : 0.0f;
            acc0[i] += fmaxf(c0.x[i], 0.0f) * w0;
            acc1[i] += fmaxf(c1.x[i], 0.0f) * w1;
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint32_t row = 2u * (uint32_t)i + (lane >> 4u);
        const uint32_t col = lane & 15u;
        const uint32_t comp = tile_c + warp * 16u + col;
        const uint32_t token0 = tile_t + row;
        const uint32_t token1 = tile_t + 16u + row;
        float out0 = acc0[i] * scale;
        float out1 = acc1[i] * scale;
        if (causal) {
            const uint32_t visible0 = (pos0 + token0 + 1u) / ratio;
            if (comp >= visible0) out0 = -INFINITY;
            const uint32_t visible1 = (pos0 + token1 + 1u) / ratio;
            if (comp >= visible1) out1 = -INFINITY;
        }
        if (token0 < n_tokens && comp < n_comp) {
            scores[(uint64_t)token0 * n_comp + comp] = out0;
        }
        if (token1 < n_tokens && comp < n_comp) {
            scores[(uint64_t)token1 * n_comp + comp] = out1;
        }
    }
#endif
}

/* Direct-load fp16-q indexer scores kernel (2026-08-11).
 *
 * a-fragments are loaded straight from the global fp16 q (row stride
 * n_head*head_dim), skipping the a_sh shared staging and the 64 per-head
 * barriers of the staged kernel. Two heads are interleaved per iteration for
 * MMA ILP; weights are preloaded to shared once. Measured 3.65x faster at
 * 98304 comps x 8192 tokens (1538 -> 421 ms, 31.3 TFLOPS) with bit-identical
 * outputs. Handles all n_tokens > 1: partial-tile fragment loads read
 * in-bounds of the pc-sized q buffer (out-of-range rows produce garbage that
 * the token guards drop) and the weights load is clamped. */
__global__ static void indexer_scores_wmma128_direct_kernel(
        float *scores,
        const __half *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700 || defined(__HIP_DEVICE_COMPILE__)
#ifdef __HIP_PLATFORM_AMD__
    namespace wmma = rocwmma;
#else
    namespace wmma = nvcuda::wmma;
#endif
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 32u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 32u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 32u * 128u; i += 256u) {
                const uint32_t r = i >> 7u;
                const uint32_t c = i & 127u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half b_sh[128 * 136];
    __shared__ float w_sh[32 * 64];
    const uint32_t lane = tid & 31u;
    float acc0[8], acc1[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) { acc0[i] = 0.0f; acc1[i] = 0.0f; }

    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 136u] = __float2half(v);
    }
    for (uint32_t i = tid; i < 32u * 64u; i += 256u) {
        const uint32_t r = i >> 6u;
        const uint32_t h = i & 63u;
        float v = 0.0f;
        if (tile_t + r < n_tokens) {
            v = weights[((uint64_t)(tile_t + r)) * n_head + h];
        }
        w_sh[r * 64u + h] = v;
    }
    __syncthreads();

    const uint32_t q_stride = n_head * head_dim;
    const uint32_t col0 = warp * 16u;
    for (uint32_t h = 0; h < n_head; h += 2u) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1, b0, b1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kb;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1, d0, d1;
        wmma::fill_fragment(c0, 0.0f);
        wmma::fill_fragment(c1, 0.0f);
        wmma::fill_fragment(d0, 0.0f);
        wmma::fill_fragment(d1, 0.0f);
        const __half *q0 = q + (uint64_t)(tile_t + 0u) * q_stride + h * head_dim;
        const __half *q1 = q + (uint64_t)(tile_t + 16u) * q_stride + h * head_dim;
        const __half *r0 = q + (uint64_t)(tile_t + 0u) * q_stride + (h + 1u) * head_dim;
        const __half *r1 = q + (uint64_t)(tile_t + 16u) * q_stride + (h + 1u) * head_dim;
#pragma unroll
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a0, q0 + k0, q_stride);
            wmma::load_matrix_sync(a1, q1 + k0, q_stride);
            wmma::load_matrix_sync(b0, r0 + k0, q_stride);
            wmma::load_matrix_sync(b1, r1 + k0, q_stride);
            wmma::load_matrix_sync(kb, b_sh + col0 * 136u + k0, 136);
            wmma::mma_sync(c0, a0, kb, c0);
            wmma::mma_sync(c1, a1, kb, c1);
            wmma::mma_sync(d0, b0, kb, d0);
            wmma::mma_sync(d1, b1, kb, d1);
        }
        const float *wh = w_sh + h;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint32_t row = 2u * (uint32_t)i + (lane >> 4u);
            acc0[i] += fmaxf(c0.x[i], 0.0f) * wh[row * 64u];
            acc1[i] += fmaxf(c1.x[i], 0.0f) * wh[(16u + row) * 64u];
            acc0[i] += fmaxf(d0.x[i], 0.0f) * wh[(row * 64u) + 1u];
            acc1[i] += fmaxf(d1.x[i], 0.0f) * wh[(16u + row) * 64u + 1u];
        }
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint32_t row = 2u * (uint32_t)i + (lane >> 4u);
        const uint32_t col = lane & 15u;
        const uint32_t comp = tile_c + warp * 16u + col;
        const uint32_t token0 = tile_t + row;
        const uint32_t token1 = tile_t + 16u + row;
        float out0 = acc0[i] * scale;
        float out1 = acc1[i] * scale;
        if (causal) {
            const uint32_t visible0 = (pos0 + token0 + 1u) / ratio;
            if (comp >= visible0) out0 = -INFINITY;
            const uint32_t visible1 = (pos0 + token1 + 1u) / ratio;
            if (comp >= visible1) out1 = -INFINITY;
        }
        if (token0 < n_tokens && comp < n_comp) {
            scores[(uint64_t)token0 * n_comp + comp] = out0;
        }
        if (token1 < n_tokens && comp < n_comp) {
            scores[(uint64_t)token1 * n_comp + comp] = out1;
        }
    }
#endif
}

__global__ static void argmax_kernel(int32_t *out_idx, const float *logits, uint32_t n_vocab) {
    enum { THREADS = 1024 };
    __shared__ float sm_val[THREADS];
    __shared__ int32_t sm_idx[THREADS];

    const uint32_t tid = threadIdx.x;
    float local_v = -INFINITY;
    int32_t local_i = 0;
    for (uint32_t i = tid; i < n_vocab; i += THREADS) {
        const float v = logits[i];
        if (v > local_v) {
            local_v = v;
            local_i = (int32_t)i;
        }
    }
    sm_val[tid] = local_v;
    sm_idx[tid] = local_i;
    __syncthreads();

    for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            const float vr = sm_val[tid + s];
            const int32_t ir = sm_idx[tid + s];
            const float vl = sm_val[tid];
            const int32_t il = sm_idx[tid];
            if ((vr > vl) || (vr == vl && ir < il)) {
                sm_val[tid] = vr;
                sm_idx[tid] = ir;
            }
        }
        __syncthreads();
    }

    if (tid == 0u) *out_idx = sm_idx[0];
}

__global__ static void indexer_topk_kernel(uint32_t *selected, const float *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *row = scores + (uint64_t)t * n_comp;
    uint32_t *sel = selected + (uint64_t)t * top_k;
    for (uint32_t k = 0; k < top_k; k++) sel[k] = 0;
    for (uint32_t c = 0; c < n_comp; c++) {
        float v = row[c];
        for (uint32_t k = 0; k < top_k; k++) {
            if ((k >= c) || v > row[sel[k]]) {
                for (uint32_t j = top_k - 1; j > k; j--) sel[j] = sel[j - 1];
                sel[k] = c;
                break;
            }
        }
    }
}

__device__ __forceinline__ static bool topk_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

/* DSpark Markov correction: select argmax(logits + W2 * W1[prev]) without
 * moving the vocabulary row back to the host. W1 and W2 are Q8_0. */
__global__ static void dspark_markov_argmax_kernel(
        unsigned long long *out_key,
        const float *logits,
        const unsigned char *w1_row,
        const unsigned char *w2,
        uint32_t vocab,
        uint32_t rank_blocks) {
    __shared__ float state[256];
    const uint32_t tid = threadIdx.x;
    if (tid < rank_blocks * 32u) {
        const uint32_t block = tid >> 5u;
        const uint32_t lane = tid & 31u;
        const unsigned char *qblock = w1_row + (uint64_t)block * 34u;
        const float scale = __half2float(*(const __half *)qblock);
        state[tid] =
            scale * (float)((const int8_t *)(qblock + 2u))[lane];
    }
    __syncthreads();

    float best_value = -INFINITY;
    uint32_t best_index = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + tid; i < vocab;
         i += gridDim.x * blockDim.x) {
        const unsigned char *row =
            w2 + (uint64_t)i * rank_blocks * 34u;
        float acc = 0.0f;
        for (uint32_t block = 0; block < rank_blocks; block++) {
            const unsigned char *qblock = row + (uint64_t)block * 34u;
            const float scale = __half2float(*(const __half *)qblock);
            const int8_t *quants = (const int8_t *)(qblock + 2u);
            float sum = 0.0f;
#pragma unroll
            for (uint32_t lane = 0; lane < 32u; lane++) {
                sum += (float)quants[lane] * state[block * 32u + lane];
            }
            acc += scale * sum;
        }
        const float value = logits[i] + acc;
        if (topk_score_better(value, i, best_value, best_index)) {
            best_value = value;
            best_index = i;
        }
    }

    __shared__ float values[256];
    __shared__ uint32_t indices[256];
    values[tid] = best_value;
    indices[tid] = best_index;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride &&
            topk_score_better(values[tid + stride], indices[tid + stride],
                              values[tid], indices[tid])) {
            values[tid] = values[tid + stride];
            indices[tid] = indices[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        const unsigned int bits = __float_as_uint(values[0]);
        const unsigned int value_key =
            (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        const unsigned long long key =
            ((unsigned long long)value_key << 32) |
            (unsigned int)(~indices[0]);
        atomicMax(out_key, key);
    }
}

__device__ __forceinline__ static uint32_t topk_float_ordered_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ static uint64_t topk_pack_key(float v, uint32_t idx) {
    return ((uint64_t)topk_float_ordered_key(v) << 32u) | (uint64_t)(0xffffffffu - idx);
}

__global__ static void indexer_topk_8192_cub_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    constexpr uint32_t BLOCK_THREADS = 512u;
    constexpr uint32_t ITEMS_PER_THREAD = 16u;
    using BlockSort = cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);

    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= BLOCK_THREADS) return;

    const float *row = scores + (uint64_t)t * n_comp;
    uint64_t keys[ITEMS_PER_THREAD];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < n_comp) {
            keys[item] = topk_pack_key(row[i], i);
        } else {
            keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
        }
    }

    BlockSort(sort_storage).SortDescending(keys);

#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < top_k) {
            selected[(uint64_t)t * top_k + i] = 0xffffffffu - (uint32_t)keys[item];
        }
    }
}

__global__ static void indexer_topk_1024_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;
    __shared__ float vals[1024];
    __shared__ uint32_t idxs[1024];

    const float *row = scores + (uint64_t)t * n_comp;
    if (tid < n_comp) {
        vals[tid] = row[tid];
        idxs[tid] = tid;
    } else {
        vals[tid] = -INFINITY;
        idxs[tid] = UINT32_MAX;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= 1024u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            uint32_t other = tid ^ j;
            if (other > tid && other < 1024u) {
                const float av = vals[tid];
                const float bv = vals[other];
                const uint32_t ai = idxs[tid];
                const uint32_t bi = idxs[other];
                const bool desc_half = (tid & k) == 0u;
                const bool swap = desc_half
                    ? topk_score_better(bv, bi, av, ai)
                    : topk_score_better(av, ai, bv, bi);
                if (swap) {
                    vals[tid] = bv;
                    idxs[tid] = bi;
                    vals[other] = av;
                    idxs[other] = ai;
                }
            }
            __syncthreads();
        }
    }

    if (tid < top_k) selected[(uint64_t)t * top_k + tid] = idxs[tid];
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_u16_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint16_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = (uint16_t)i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT16_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = (uint16_t)bi;
                        vals[other] = av;
                        idxs[other] = (uint16_t)ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_chunk_pow2_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t chunk = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t chunk_start = chunk * SORT_N;
    if (chunk_start >= n_comp) return;
    const uint32_t chunk_n = n_comp - chunk_start < SORT_N ? n_comp - chunk_start : SORT_N;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) {
            vals[i] = row[chunk_start + i];
            idxs[i] = chunk_start + i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        out[i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_merge_pow2_kernel(
        uint32_t *selected,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_tree_merge_pow2_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride) {
    uint32_t t = blockIdx.x;
    uint32_t group = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t set0 = group * merge_group;
    if (set0 >= n_sets) return;
    uint32_t set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const uint32_t candidate_count = set_count * top_k;

    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride + set0 * top_k;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *dst = out + (uint64_t)t * out_stride + group * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        dst[i] = idxs[i];
    }
}

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}

__global__ static void topk_mask_kernel(float *mask, const uint32_t *topk, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_comp;
    if (gid >= n) return;
    uint32_t t = gid / n_comp;
    uint32_t c = gid - (uint64_t)t * n_comp;
    float v = -INFINITY;
    for (uint32_t k = 0; k < top_k; k++) {
        if (topk[(uint64_t)t * top_k + k] == c) {
            v = 0.0f;
            break;
        }
    }
    mask[gid] = v;
}

static int indexer_scores_launch(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        uint32_t                causal) {
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
        return 0;
    }
    if (causal && ratio == 0) return 0;
    if (n_tokens == 1u && head_dim == 128u && n_head == 64u) {
        indexer_score_one_direct_kernel<<<n_comp, 128>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, pos0, ratio,
                                                         scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer score one direct launch");
    }
    if (!g_quality_mode && head_dim == 128u && n_head == 64u) {
        dim3 grid((n_comp + 127u) / 128u, (n_tokens + 31u) / 32u, 1);
        indexer_scores_wmma128_staged_kernel_t<float><<<grid, 256>>>((float *)scores->ptr,
                                                              (const float *)q->ptr,
                                                              (const float *)weights->ptr,
                                                              (const float *)index_comp->ptr,
                                                              n_comp, n_tokens, pos0, n_head,
                                                              head_dim, ratio, scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer scores wmma128 launch");
    }
    dim3 grid(n_comp, n_tokens, 1);
    indexer_scores_kernel<<<grid, 256>>>((float *)scores->ptr,
                                         (const float *)q->ptr,
                                         (const float *)weights->ptr,
                                         (const float *)index_comp->ptr,
                                         n_comp, n_tokens, pos0, n_head,
                                         head_dim, ratio, scale, causal ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "indexer scores launch");
}

/* fp16-q variant of the batch scores launch: the QAT step has already
 * converted the indexer q to fp16 (bit-identical MMA inputs; the in-kernel
 * __float2half is skipped and the a_sh staging reads half the bytes).
 * Falls back to the fp32 one-token kernel when n_tokens == 1. */
static int indexer_scores_f16q_launch(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q16,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    if (!scores || !q16 || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        q16->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(__half) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
        return 0;
    }
    if (ratio == 0) return 0;
    if (n_tokens == 1u && head_dim == 128u && n_head == 64u) {
        indexer_score_one_direct_kernel<<<n_comp, 128>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, pos0, ratio,
                                                         scale, 1);
        return cuda_ok(cudaGetLastError(), "indexer score one direct f16q launch");
    }
    if (n_tokens > 1u && !g_quality_mode && head_dim == 128u && n_head == 64u) {
        dim3 grid((n_comp + 127u) / 128u, (n_tokens + 31u) / 32u, 1);
        /* Direct-load kernel: a-fragments from global fp16 q, no a_sh
         * staging, no per-head barriers, two heads interleaved. Measured
         * 3.65x faster than the staged kernel at 98304 comps x 8192
         * tokens (1538 -> 421 ms) with bit-identical outputs. The direct
         * fragment loads read the full 32-token tile, which stays inside
         * the pc-sized q buffer for any n_tokens <= pc (partial-tile rows
         * compute garbage that the token guards drop); the weights load is
         * clamped. */
        indexer_scores_wmma128_direct_kernel<<<grid, 256>>>((float *)scores->ptr,
                                                           (const __half *)q16->ptr,
                                                           (const float *)weights->ptr,
                                                           (const float *)index_comp->ptr,
                                                           n_comp, n_tokens, pos0, n_head,
                                                           head_dim, ratio, scale, 1);
        return cuda_ok(cudaGetLastError(), "indexer scores wmma128 direct f16q launch");
    }
    return 0;
}

extern "C" int ds4_gpu_indexer_scores_decode_batch_f16_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q16,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_f16q_launch(scores, q16, q, weights, index_comp,
                                      n_comp, n_tokens, pos0, n_head, head_dim,
                                      ratio, scale);
}

extern "C" int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, 1, 0,
                                 n_head, head_dim, 1, scale, 0);
}

extern "C" int ds4_gpu_indexer_scores_prefill_f16_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q16,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_f16q_launch(scores, q16, q, weights, index_comp,
                                      n_comp, n_tokens, 0, n_head, head_dim,
                                      ratio, scale);
}

extern "C" int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, 0,
                                 n_head, head_dim, ratio, scale, 1);
}

extern "C" int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, pos0,
                                 n_head, head_dim, ratio, scale, 1);
}

/* Radix-sort top-k over 8192-row chunks (replaces the 4096-row bitonic tree
 * for n_comp > 8192).  Each chunk keeps top_k candidates; groups of
 * 8192/top_k chunks are merged per block (each merge sorts <= 8192 packed
 * (score, index) keys), with a final single-block merge.  CUB BlockRadixSort
 * replaces the 144-pass bitonic network: measured 1.3-1.4x faster at
 * 16K-200K comps (4096 tokens), bit-exact vs the bitonic tree. */
template <uint32_t CHUNK_N>
__global__ static void indexer_topk_chunk_cub_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride) {
    constexpr uint32_t THREADS = 512u;
    constexpr uint32_t IPT = CHUNK_N / THREADS;
    using BlockSort = cub::BlockRadixSort<uint64_t, THREADS, IPT>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);
    const uint32_t t = blockIdx.x;
    const uint32_t chunk = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= THREADS) return;
    const uint32_t chunk_start = chunk * CHUNK_N;
    if (chunk_start >= n_comp) return;
    const uint32_t chunk_n = n_comp - chunk_start < CHUNK_N ? n_comp - chunk_start : CHUNK_N;
    const float *row = scores + (uint64_t)t * n_comp;
    uint64_t keys[IPT];
#pragma unroll
    for (uint32_t item = 0; item < IPT; item++) {
        const uint32_t i = tid * IPT + item;
        if (i < chunk_n) keys[item] = topk_pack_key(row[chunk_start + i], chunk_start + i);
        else keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
    }
    BlockSort(sort_storage).SortDescending(keys);
    uint32_t *out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
#pragma unroll
    for (uint32_t item = 0; item < IPT; item++) {
        const uint32_t i = tid * IPT + item;
        if (i < top_k) out[i] = 0xffffffffu - (uint32_t)keys[item];
    }
}

template <uint32_t CAP>
__global__ static void indexer_topk_tree_merge_cub_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride) {
    constexpr uint32_t THREADS = 512u;
    constexpr uint32_t IPT = CAP / THREADS;
    using BlockSort = cub::BlockRadixSort<uint64_t, THREADS, IPT>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);
    const uint32_t t = blockIdx.x;
    const uint32_t group = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= THREADS) return;
    const uint32_t set0 = group * merge_group;
    if (set0 >= n_sets) return;
    uint32_t set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const uint32_t candidate_count = set_count * top_k;
    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride + set0 * top_k;
    uint64_t keys[IPT];
#pragma unroll
    for (uint32_t item = 0; item < IPT; item++) {
        const uint32_t i = tid * IPT + item;
        if (i < candidate_count) {
            const uint32_t idx = cand[i];
            keys[item] = (idx < n_comp) ? topk_pack_key(row[idx], idx)
                                        : topk_pack_key(-INFINITY, UINT32_MAX);
        } else {
            keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
        }
    }
    BlockSort(sort_storage).SortDescending(keys);
    uint32_t *dst = out + (uint64_t)t * out_stride + group * top_k;
#pragma unroll
    for (uint32_t item = 0; item < IPT; item++) {
        const uint32_t i = tid * IPT + item;
        if (i < top_k) dst[i] = 0xffffffffu - (uint32_t)keys[item];
    }
}

template <uint32_t CAP>
__global__ static void indexer_topk_final_merge_cub_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride) {
    constexpr uint32_t THREADS = 512u;
    constexpr uint32_t IPT = CAP / THREADS;
    using BlockSort = cub::BlockRadixSort<uint64_t, THREADS, IPT>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= THREADS) return;
    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride;
    uint64_t keys[IPT];
#pragma unroll
    for (uint32_t item = 0; item < IPT; item++) {
        const uint32_t i = tid * IPT + item;
        if (i < candidate_count) {
            const uint32_t idx = cand[i];
            keys[item] = (idx < n_comp) ? topk_pack_key(row[idx], idx)
                                        : topk_pack_key(-INFINITY, UINT32_MAX);
        } else {
            keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
        }
    }
    BlockSort(sort_storage).SortDescending(keys);
#pragma unroll
    for (uint32_t item = 0; item < IPT; item++) {
        const uint32_t i = tid * IPT + item;
        if (i < top_k) out[(uint64_t)t * top_k + i] = 0xffffffffu - (uint32_t)keys[item];
    }
}

extern "C" int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    if (top_k == 512u && n_comp <= 1024u) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 1024 launch");
    }
    if (top_k == 512u && n_comp <= 2048u) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048 launch");
    }
    if (top_k == 512u && n_comp <= 4096u) {
        if (n_comp == 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 4096 cub launch");
                }
            }
        }
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096 launch");
    }
    if (top_k == 512u && n_comp <= 8192u) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 8192 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192 launch");
    }
    if (top_k == 1024u && n_comp <= 1024u) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 1024x1024 launch");
    }
    if (top_k == 1024u && n_comp <= 2048u) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048x1024 launch");
    }
    if (top_k == 1024u && n_comp <= 4096u) {
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096x1024 launch");
    }
    if (top_k == 1024u && n_comp <= 8192u) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 8192x1024 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192x1024 launch");
    }
    if (top_k == 2048u && n_comp <= 4096u) {
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096x2048 launch");
    }
    if (top_k == 2048u && n_comp <= 8192u) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 8192x2048 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192x2048 launch");
    }
    if (top_k == 512u || top_k == 1024u || top_k == 2048u) {
        /* CUB radix tree over 8192-row chunks when the device allows the
         * 64 KiB BlockRadixSort temp storage; otherwise fall back to the
         * 4096-row bitonic tree below. */
        using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
        const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
        int dev = 0;
        int max_optin_smem = 0;
        cudaError_t attr_err = cudaGetDevice(&dev);
        if (attr_err == cudaSuccess) {
            attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                              cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                              dev);
        }
        if (attr_err == cudaSuccess && max_optin_smem >= smem) {
            const uint32_t chunk_n = 8192u;
            const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
            const uint32_t merge_group = chunk_n / top_k;
            const uint64_t candidate_stride64 = (uint64_t)n_chunks * top_k;
            if (candidate_stride64 > UINT32_MAX) return 0;
            const uint32_t candidate_stride = (uint32_t)candidate_stride64;
            uint32_t n_sets = n_chunks;
            uint64_t scratch_u32_per_token = candidate_stride;
            while (n_sets > merge_group) {
                n_sets = (n_sets + merge_group - 1u) / merge_group;
                scratch_u32_per_token += (uint64_t)n_sets * top_k;
            }
            if (scratch_u32_per_token > UINT64_MAX / n_tokens / sizeof(uint32_t)) return 0;
            const uint64_t tmp_bytes = (uint64_t)n_tokens * scratch_u32_per_token * sizeof(uint32_t);
            uint32_t *scratch = (uint32_t *)cuda_tmp_alloc(tmp_bytes, "indexer topk tree");
            if (!scratch) return 0;

            attr_err = cudaFuncSetAttribute(indexer_topk_chunk_cub_kernel<8192>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            smem);
            if (attr_err != cudaSuccess) return 0;
            attr_err = cudaFuncSetAttribute(indexer_topk_tree_merge_cub_kernel<8192>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            smem);
            if (attr_err != cudaSuccess) return 0;
            attr_err = cudaFuncSetAttribute(indexer_topk_final_merge_cub_kernel<8192>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            smem);
            if (attr_err != cudaSuccess) return 0;

            uint32_t *cur = scratch;
            n_sets = n_chunks;
            uint32_t cur_stride = candidate_stride;
            dim3 grid_chunks(n_tokens, n_chunks, 1);
            indexer_topk_chunk_cub_kernel<8192><<<grid_chunks, 512, smem>>>(
                    cur,
                    (const float *)scores->ptr,
                    n_comp,
                    n_tokens,
                    top_k,
                    candidate_stride);
            if (!cuda_ok(cudaGetLastError(), "indexer topk cub chunk launch")) return 0;

            while (n_sets > merge_group) {
                const uint32_t next_sets = (n_sets + merge_group - 1u) / merge_group;
                const uint32_t next_stride = next_sets * top_k;
                uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
                dim3 grid_merge(n_tokens, next_sets, 1);
                indexer_topk_tree_merge_cub_kernel<8192><<<grid_merge, 512, smem>>>(
                        next,
                        cur,
                        (const float *)scores->ptr,
                        n_comp,
                        n_tokens,
                        top_k,
                        n_sets,
                        merge_group,
                        cur_stride,
                        next_stride);
                if (!cuda_ok(cudaGetLastError(), "indexer topk cub tree merge launch")) return 0;
                cur = next;
                n_sets = next_sets;
                cur_stride = next_stride;
            }

            indexer_topk_final_merge_cub_kernel<8192><<<n_tokens, 512, smem>>>(
                    (uint32_t *)selected->ptr,
                    cur,
                    (const float *)scores->ptr,
                    n_comp,
                    n_tokens,
                    top_k,
                    n_sets * top_k,
                    cur_stride);
            return cuda_ok(cudaGetLastError(), "indexer topk cub final merge launch");
        }

        /* Fallback: 4096-row bitonic tree (devices without 64 KiB opt-in). */
        const uint32_t chunk_n = 4096u;
        const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
        const uint32_t merge_group = chunk_n / top_k;
        const uint64_t candidate_stride64 = (uint64_t)n_chunks * top_k;
        if (candidate_stride64 > UINT32_MAX) return 0;
        const uint32_t candidate_stride = (uint32_t)candidate_stride64;
        uint32_t n_sets = n_chunks;
        uint64_t scratch_u32_per_token = candidate_stride;
        while (n_sets > merge_group) {
            n_sets = (n_sets + merge_group - 1u) / merge_group;
            scratch_u32_per_token += (uint64_t)n_sets * top_k;
        }
        if (scratch_u32_per_token > UINT64_MAX / n_tokens / sizeof(uint32_t)) return 0;
        const uint64_t tmp_bytes = (uint64_t)n_tokens * scratch_u32_per_token * sizeof(uint32_t);
        uint32_t *scratch = (uint32_t *)cuda_tmp_alloc(tmp_bytes, "indexer topk tree");
        if (!scratch) return 0;

        uint32_t *cur = scratch;
        n_sets = n_chunks;
        uint32_t cur_stride = candidate_stride;
        dim3 grid_chunks(n_tokens, n_chunks, 1);
        indexer_topk_chunk_pow2_kernel<4096><<<grid_chunks, 1024>>>(cur,
                                                                    (const float *)scores->ptr,
                                                                    n_comp,
                                                                    n_tokens,
                                                                    top_k,
                                                                    candidate_stride);
        if (!cuda_ok(cudaGetLastError(), "indexer topk chunk launch")) return 0;

        while (n_sets > merge_group) {
            const uint32_t next_sets = (n_sets + merge_group - 1u) / merge_group;
            const uint32_t next_stride = next_sets * top_k;
            uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
            dim3 grid_merge(n_tokens, next_sets, 1);
            indexer_topk_tree_merge_pow2_kernel<4096><<<grid_merge, 1024>>>(
                    next,
                    cur,
                    (const float *)scores->ptr,
                    n_comp,
                    n_tokens,
                    top_k,
                    n_sets,
                    merge_group,
                    cur_stride,
                    next_stride);
            if (!cuda_ok(cudaGetLastError(), "indexer topk tree merge launch")) return 0;
            cur = next;
            n_sets = next_sets;
            cur_stride = next_stride;
        }

        indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                                 cur,
                                                                 (const float *)scores->ptr,
                                                                 n_comp,
                                                                 n_tokens,
                                                                 top_k,
                                                                 n_sets * top_k,
                                                                 cur_stride);
        return cuda_ok(cudaGetLastError(), "indexer topk tree final launch");
    }
    indexer_topk_kernel<<<n_tokens, 1>>>((uint32_t *)selected->ptr,
                                         (const float *)scores->ptr,
                                         n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "indexer topk launch");
}

extern "C" int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab) {
    uint64_t logits_bytes = 0;
    if (!out_idx || !logits || n_vocab == 0u ||
        out_idx->bytes < sizeof(int32_t) ||
        !cuda_u64_mul3_checked(n_vocab, 1u, sizeof(float), &logits_bytes) ||
        logits->bytes < logits_bytes) {
        return 0;
    }
    argmax_kernel<<<1, 1024>>>((int32_t *)out_idx->ptr,
                               (const float *)logits->ptr,
                               n_vocab);
    return cuda_ok(cudaGetLastError(), "argmax launch");
}

extern "C" int ds4_gpu_dspark_markov_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits_row,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              w1_offset,
        uint64_t              w2_offset,
        uint32_t              prev_token,
        uint32_t              vocab,
        uint32_t              rank) {
    if (!out_idx || !logits_row || !model_map || vocab == 0 ||
        rank == 0 || (rank & 31u) != 0u || rank > 256u ||
        out_idx->bytes < sizeof(unsigned long long) ||
        logits_row->bytes < (uint64_t)vocab * sizeof(float)) {
        return 0;
    }
    const uint32_t rank_blocks = rank / 32u;
    const uint64_t row_bytes = (uint64_t)rank_blocks * 34u;
    if (prev_token > UINT64_MAX / row_bytes ||
        vocab > UINT64_MAX / row_bytes) {
        return 0;
    }
    const uint64_t w1_row_offset = (uint64_t)prev_token * row_bytes;
    const uint64_t w2_bytes = (uint64_t)vocab * row_bytes;
    if (w1_offset > model_size ||
        w1_row_offset > model_size - w1_offset ||
        row_bytes > model_size - w1_offset - w1_row_offset ||
        w2_offset > model_size || w2_bytes > model_size - w2_offset) {
        return 0;
    }
    const unsigned char *w1_row =
        (const unsigned char *)cuda_model_range_ptr(
            model_map,
            w1_offset + w1_row_offset,
            row_bytes,
            "markov_w1_row");
    const unsigned char *w2 =
        (const unsigned char *)cuda_model_range_ptr(
            model_map, w2_offset, w2_bytes, "markov_w2");
    if (!w1_row || !w2) return 0;

    if (!cuda_ok(cudaMemsetAsync(out_idx->ptr, 0,
                                 sizeof(unsigned long long)),
                 "DSpark markov argmax clear")) {
        return 0;
    }
    dspark_markov_argmax_kernel<<<128, 256>>>(
            (unsigned long long *)out_idx->ptr,
            (const float *)logits_row->ptr,
            w1_row,
            w2,
            vocab,
            rank_blocks);
    return cuda_ok(cudaGetLastError(), "DSpark markov argmax launch");
}

extern "C" int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_tokens * n_comp;
    uint64_t nk = (uint64_t)n_tokens * top_k;
    uint64_t blocks = ((n > nk ? n : nk) + 255) / 256;
    topk_mask_kernel<<<blocks, 256>>>((float *)mask->ptr,
                                      (const uint32_t *)topk->ptr,
                                      n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "topk mask launch");
}

extern "C" int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<<<n_rows, 128>>>((float *)x->ptr, NULL, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4 launch");
}

extern "C" int ds4_gpu_dsv4_indexer_qat_f16_tensor(ds4_gpu_tensor *x, ds4_gpu_tensor *x16, uint32_t n_rows, uint32_t head_dim) {
    if (!x || !x16 || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float) ||
        x16->bytes < (uint64_t)n_rows * head_dim * sizeof(__half)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<<<n_rows, 128>>>((float *)x->ptr, (__half *)x16->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4 f16 launch");
}
