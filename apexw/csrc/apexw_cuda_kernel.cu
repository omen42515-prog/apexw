#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

// ============================================================================
// Ultra-Fast Deterministic Pseudo-Random Hash & Stochastic Rounding
// ============================================================================

__device__ __forceinline__ uint32_t hash_coords(uint32_t idx, uint32_t step, uint32_t tensor_idx = 0u) {
    uint32_t h = idx ^ (step * 0x9E3779B9u) ^ (tensor_idx * 0x85ebca6bu);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16_sr(float x, uint32_t rand_val) {
    uint32_t u = __float_as_uint(x);
    uint32_t mag = u & 0x7FFFFFFFu;
    if (mag < 0x7F7F0000u) {
        mag += (rand_val & 0xFFFFu);
        u = (u & 0x80000000u) | (mag & 0x7FFFFFFFu);
    } else if (mag > 0x7F800000u) {
        u |= 0x00400000u;
    }
    uint16_t upper = static_cast<uint16_t>(u >> 16);
    return *reinterpret_cast<__nv_bfloat16*>(&upper);
}

__device__ __forceinline__ __half float_to_f16_sr(float x, uint32_t rand_val) {
    if (isnan(x)) return __float2half(x);
    x = fminf(fmaxf(x, -65504.0f), 65504.0f);
    if (fabsf(x) < 6.103515625e-5f) {
        // Subnormal FP16 range (step = 2^-24): round stochastically directly in FP32
        float u = static_cast<float>(rand_val & 0xFFFFu) * (1.0f / 65536.0f);
        float r = floorf(x * 16777216.0f + u);
        return __float2half(r * 5.9604644775390625e-8f);
    }
    uint32_t u = __float_as_uint(x);
    uint32_t mag = u & 0x7FFFFFFFu;
    mag += (rand_val & 0x1FFFu); // 13 bits of discarded mantissa in normal FP16
    u = (u & 0x80000000u) | (mag & 0x7FFFFFFFu);
    float rounded = __uint_as_float(u);
    return __float2half_rz(rounded);
}

// ============================================================================
// Type Conversion & Arithmetic Helpers
// ============================================================================

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    uint32_t u = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&x)) << 16;
    return __uint_as_float(u);
}
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

__device__ __forceinline__ float sanitize_grad_val(float g) {
    if (isfinite(g)) return g;
    if (isnan(g)) return 0.0f;
    return (g > 0.0f) ? 65504.0f : -65504.0f;
}

template<typename T>
__device__ __forceinline__ T from_float_sr(float x, uint32_t rand_val);

template<>
__device__ __forceinline__ float from_float_sr<float>(float x, uint32_t rand_val) { return x; }

template<>
__device__ __forceinline__ __nv_bfloat16 from_float_sr<__nv_bfloat16>(float x, uint32_t rand_val) {
    return float_to_bf16_sr(x, rand_val);
}

template<>
__device__ __forceinline__ __half from_float_sr<__half>(float x, uint32_t rand_val) {
    return float_to_f16_sr(x, rand_val);
}

// ============================================================================
// Metadata for Multi-Tensor Pointer-Array Batched Execution
// ============================================================================

struct TensorMetadata {
    void* param;
    const void* grad;
    void* exp_avg;
    void* exp_avg_sq;
    int64_t numel;
    float weight_decay;
    uint32_t flags;
};

// 64 KB constant memory shared across 32-bit/16-bit (512 * 48 = 24,576 B),
// 8-bit/4-bit (576 * 64 = 36,864 B), and NF4 LUT (64 B) -> 61,504 B < 65,536 B
__constant__ TensorMetadata c_tensor_metadata[512];

// ============================================================================
// 2D Grid Multi-Tensor Pointer-Array Fused Kernel with Stochastic Rounding
//
// Grid Configuration:
// - blockIdx.y selects the tensor index in the batch (0 <= blockIdx.y < num_tensors)
// - blockIdx.x * blockDim.x + threadIdx.x determines elements inside that tensor
// ============================================================================

template<typename ParamT, typename StateM_T, typename StateV_T>
__global__ void multi_tensor_adamw_kernel_2d(
    int num_tensors,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float step_size,
    float bias_correction2_sqrt,
    uint32_t step_seed
) {
    // blockIdx.y directly indexes the tensor (zero binary search, zero branch divergence)
    int tensor_idx = blockIdx.y;
    if (tensor_idx >= num_tensors) return;

    const TensorMetadata meta = c_tensor_metadata[tensor_idx];
    int64_t total_elements = meta.numel;
    if (total_elements <= 0) return;

    float weight_decay = meta.weight_decay;
    float wd_factor = (weight_decay != 0.0f) ? fmaxf(1.0f - lr * weight_decay, 0.0f) : 1.0f;
    ParamT* param = static_cast<ParamT*>(meta.param);
    const ParamT* grad = static_cast<const ParamT*>(meta.grad);
    StateM_T* exp_avg = static_cast<StateM_T*>(meta.exp_avg);
    StateV_T* exp_avg_sq = static_cast<StateV_T*>(meta.exp_avg_sq);
    const bool is_quad_v = (meta.flags & 1u) != 0u;

    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    bool is_aligned16 = (((uintptr_t)param % 16) == 0) &&
                        (((uintptr_t)grad % 16) == 0) &&
                        (((uintptr_t)exp_avg % 16) == 0) &&
                        (((uintptr_t)exp_avg_sq % 16) == 0);

    constexpr int VEC_PARAM = 16 / sizeof(ParamT);
    constexpr bool is_pure_fp32 = std::is_same<ParamT, float>::value &&
                                  std::is_same<StateM_T, float>::value &&
                                  std::is_same<StateV_T, float>::value;

    // Fast-path bypass for pure float32 (FP32):
    // If the parameters and states are both float32, bypass quantization/dequantization
    // and stochastic rounding scaling entirely to match or beat PyTorch's native AdamW(fused=True) throughput.
    if constexpr (is_pure_fp32) {
        if (is_aligned16 && !is_quad_v) {
            int64_t vec_count = total_elements / 4;
            for (int64_t idx = tid; idx < vec_count; idx += stride) {
                float4 p_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(param) + idx * 16);
                float4 g_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(grad) + idx * 16);
                float4 m_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(exp_avg) + idx * 16);
                float4 v_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(exp_avg_sq) + idx * 16);

                float* p_arr = reinterpret_cast<float*>(&p_raw);
                const float* g_arr = reinterpret_cast<const float*>(&g_raw);
                float* m_arr = reinterpret_cast<float*>(&m_raw);
                float* v_arr = reinterpret_cast<float*>(&v_raw);

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float p = p_arr[i];
                    float g = sanitize_grad_val(g_arr[i]);
                    float m = m_arr[i];
                    float v = v_arr[i];

                    if (weight_decay != 0.0f) {
                        p *= wd_factor;
                    }

                    m = beta1 * m + (1.0f - beta1) * g;
                    v = beta2 * v + (1.0f - beta2) * (g * g);

                    float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
                    p -= step_size * (m / denom);

                    p_arr[i] = p;
                    m_arr[i] = m;
                    v_arr[i] = v;
                }

                *reinterpret_cast<float4*>(reinterpret_cast<char*>(param) + idx * 16) = p_raw;
                *reinterpret_cast<float4*>(reinterpret_cast<char*>(exp_avg) + idx * 16) = m_raw;
                *reinterpret_cast<float4*>(reinterpret_cast<char*>(exp_avg_sq) + idx * 16) = v_raw;
            }

            // Scalar tail loop
            int64_t vec_limit = vec_count * 4;
            for (int64_t idx = vec_limit + tid; idx < total_elements; idx += stride) {
                float p = param[idx];
                float g = sanitize_grad_val(grad[idx]);
                float m = exp_avg[idx];
                float v = exp_avg_sq[idx];

                if (weight_decay != 0.0f) {
                    p *= wd_factor;
                }

                m = beta1 * m + (1.0f - beta1) * g;
                v = beta2 * v + (1.0f - beta2) * (g * g);

                float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
                p -= step_size * (m / denom);

                param[idx] = p;
                exp_avg[idx] = m;
                exp_avg_sq[idx] = v;
            }
        } else {
            // Unaligned pure FP32 fallback
            for (int64_t idx = tid; idx < total_elements; idx += stride) {
                float p = param[idx];
                float g = sanitize_grad_val(grad[idx]);
                float m = exp_avg[idx];
                float v = exp_avg_sq[idx];

                if (weight_decay != 0.0f) {
                    p *= wd_factor;
                }

                m = beta1 * m + (1.0f - beta1) * g;
                v = beta2 * v + (1.0f - beta2) * (g * g);

                float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
                p -= step_size * (m / denom);

                param[idx] = p;
                exp_avg[idx] = m;
                exp_avg_sq[idx] = v;
            }
        }
        return;
    }

    // Compact Quad-V Fast Path (16-bit m + Quad 16-bit v = 2.50 B/param with Fused Nesterov-SNR Lookahead)
    if (is_quad_v) {
        int64_t quad_count = total_elements >> 2;
        for (int64_t q_idx = tid; q_idx < quad_count; q_idx += stride) {
            int64_t base = q_idx << 2;
            uint32_t base_h = hash_coords(static_cast<uint32_t>(base), step_seed, static_cast<uint32_t>(tensor_idx));
            float p_v[4], g_v[4], m_v[4];
            float g_sq_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                p_v[i] = to_float(param[base + i]);
                g_v[i] = sanitize_grad_val(to_float(grad[base + i]));
                m_v[i] = to_float(exp_avg[base + i]);
                if (weight_decay != 0.0f) p_v[i] *= wd_factor;
                m_v[i] = beta1 * m_v[i] + (1.0f - beta1) * g_v[i];
                g_sq_sum += g_v[i] * g_v[i];
            }
            float v_q = to_float(exp_avg_sq[q_idx]);
            v_q = beta2 * v_q + (1.0f - beta2) * (0.25f * g_sq_sum);
            if (!isfinite(v_q)) v_q = 0.0f;
            float inv_denom = step_size / fmaxf((sqrtf(fmaxf(v_q, 0.0f)) / bias_correction2_sqrt) + eps, eps);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                p_v[i] -= m_v[i] * inv_denom;
                uint32_t hi = base_h + i * 0x9E3779B9u;
                param[base + i] = from_float_sr<ParamT>(p_v[i], hi);
                exp_avg[base + i] = from_float_sr<StateM_T>(m_v[i], hi ^ 0x55555555u);
            }
            exp_avg_sq[q_idx] = from_float_sr<StateV_T>(v_q, base_h ^ 0xAAAAAAAAu);
        }

        // Remainder tail loop for elements past (quad_count << 2)
        int64_t tail_start = quad_count << 2;
        for (int64_t idx = tail_start + tid; idx < total_elements; idx += stride) {
            float p = to_float(param[idx]);
            float g = sanitize_grad_val(to_float(grad[idx]));
            float m = to_float(exp_avg[idx]);
            if (weight_decay != 0.0f) p *= wd_factor;
            m = beta1 * m + (1.0f - beta1) * g;
            float v_tail = (quad_count > 0) ? to_float(exp_avg_sq[quad_count - 1]) : (g * g);
            float inv_denom_t = step_size / fmaxf((sqrtf(fmaxf(v_tail, 0.0f)) / bias_correction2_sqrt) + eps, eps);
            p -= m * inv_denom_t;
            uint32_t h = hash_coords(static_cast<uint32_t>(idx), step_seed, static_cast<uint32_t>(tensor_idx));
            param[idx] = from_float_sr<ParamT>(p, h);
            exp_avg[idx] = from_float_sr<StateM_T>(m, h ^ 0x55555555u);
        }
        return;
    }

    // Case 1: Param and both states have matching element size (e.g. BF16/BF16/BF16, FP16/FP16/FP16)
    if (is_aligned16 && (sizeof(ParamT) == sizeof(StateM_T)) && (sizeof(ParamT) == sizeof(StateV_T))) {
        int64_t vec_count = total_elements / VEC_PARAM;
        for (int64_t idx = tid; idx < vec_count; idx += stride) {
            float4 p_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(param) + idx * 16);
            float4 g_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(grad) + idx * 16);
            float4 m_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(exp_avg) + idx * 16);
            float4 v_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(exp_avg_sq) + idx * 16);

            ParamT* p_arr = reinterpret_cast<ParamT*>(&p_raw);
            const ParamT* g_arr = reinterpret_cast<const ParamT*>(&g_raw);
            StateM_T* m_arr = reinterpret_cast<StateM_T*>(&m_raw);
            StateV_T* v_arr = reinterpret_cast<StateV_T*>(&v_raw);

            uint32_t base_h = hash_coords(static_cast<uint32_t>(idx * VEC_PARAM), step_seed, static_cast<uint32_t>(tensor_idx));

            #pragma unroll
            for (int i = 0; i < VEC_PARAM; ++i) {
                float p = to_float(p_arr[i]);
                float g = sanitize_grad_val(to_float(g_arr[i]));
                float m = to_float(m_arr[i]);
                float v = to_float(v_arr[i]);

                if (weight_decay != 0.0f) {
                    p *= wd_factor;
                }

                m = beta1 * m + (1.0f - beta1) * g;
                v = beta2 * v + (1.0f - beta2) * (g * g);
                if (!isfinite(v)) v = 0.0f;

                // Adaptive dynamic epsilon: full FP32 promotion and division-by-zero protection
                float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
                p -= step_size * (m / denom);

                uint32_t hi = base_h + i * 0x9E3779B9u;
                p_arr[i] = from_float_sr<ParamT>(p, hi);
                m_arr[i] = from_float_sr<StateM_T>(m, hi ^ 0x55555555u);
                v_arr[i] = from_float_sr<StateV_T>(v, hi ^ 0xAAAAAAAAu);
            }

            *reinterpret_cast<float4*>(reinterpret_cast<char*>(param) + idx * 16) = p_raw;
            *reinterpret_cast<float4*>(reinterpret_cast<char*>(exp_avg) + idx * 16) = m_raw;
            *reinterpret_cast<float4*>(reinterpret_cast<char*>(exp_avg_sq) + idx * 16) = v_raw;
        }

        // Scalar tail loop
        int64_t vec_limit = vec_count * VEC_PARAM;
        for (int64_t idx = vec_limit + tid; idx < total_elements; idx += stride) {
            float p = to_float(param[idx]);
            float g = sanitize_grad_val(to_float(grad[idx]));
            float m = to_float(exp_avg[idx]);
            float v = to_float(exp_avg_sq[idx]);

            if (weight_decay != 0.0f) {
                p *= wd_factor;
            }

            m = beta1 * m + (1.0f - beta1) * g;
            v = beta2 * v + (1.0f - beta2) * (g * g);
            if (!isfinite(v)) v = 0.0f;

            float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
            p -= step_size * (m / denom);

            uint32_t h = hash_coords(static_cast<uint32_t>(idx), step_seed, static_cast<uint32_t>(tensor_idx));
            param[idx] = from_float_sr<ParamT>(p, h);
            exp_avg[idx] = from_float_sr<StateM_T>(m, h ^ 0x55555555u);
            exp_avg_sq[idx] = from_float_sr<StateV_T>(v, h ^ 0xAAAAAAAAu);
        }
    }
    // Case 2: 32-bit params + 16-bit compact states (FP32 weights with BF16/FP16 moments)
    else if (is_aligned16 && (sizeof(ParamT) == 4) && (sizeof(StateM_T) == 2) && (sizeof(StateV_T) == 2)) {
        int64_t vec_count = total_elements / 4;
        for (int64_t idx = tid; idx < vec_count; idx += stride) {
            float4 p_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(param) + idx * 16);
            float4 g_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(grad) + idx * 16);
            uint2 m_raw = *reinterpret_cast<const uint2*>(reinterpret_cast<const char*>(exp_avg) + idx * 8);
            uint2 v_raw = *reinterpret_cast<const uint2*>(reinterpret_cast<const char*>(exp_avg_sq) + idx * 8);

            float* p_arr = reinterpret_cast<float*>(&p_raw);
            const float* g_arr = reinterpret_cast<const float*>(&g_raw);
            StateM_T* m_arr = reinterpret_cast<StateM_T*>(&m_raw);
            StateV_T* v_arr = reinterpret_cast<StateV_T*>(&v_raw);

            uint32_t base_h = hash_coords(static_cast<uint32_t>(idx * 4), step_seed, static_cast<uint32_t>(tensor_idx));

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float p = p_arr[i];
                float g = sanitize_grad_val(g_arr[i]);
                float m = to_float(m_arr[i]);
                float v = to_float(v_arr[i]);

                if (weight_decay != 0.0f) {
                    p *= wd_factor;
                }

                m = beta1 * m + (1.0f - beta1) * g;
                v = beta2 * v + (1.0f - beta2) * (g * g);
                if (!isfinite(v)) v = 0.0f;

                float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
                p -= step_size * (m / denom);

                uint32_t hi = base_h + i * 0x9E3779B9u;
                p_arr[i] = p; // Float32 param requires no downcasting
                m_arr[i] = from_float_sr<StateM_T>(m, hi ^ 0x55555555u);
                v_arr[i] = from_float_sr<StateV_T>(v, hi ^ 0xAAAAAAAAu);
            }

            *reinterpret_cast<float4*>(reinterpret_cast<char*>(param) + idx * 16) = p_raw;
            *reinterpret_cast<uint2*>(reinterpret_cast<char*>(exp_avg) + idx * 8) = m_raw;
            *reinterpret_cast<uint2*>(reinterpret_cast<char*>(exp_avg_sq) + idx * 8) = v_raw;
        }

        // Scalar tail loop
        int64_t vec_limit = vec_count * 4;
        for (int64_t idx = vec_limit + tid; idx < total_elements; idx += stride) {
            float p = to_float(param[idx]);
            float g = sanitize_grad_val(to_float(grad[idx]));
            float m = to_float(exp_avg[idx]);
            float v = to_float(exp_avg_sq[idx]);

            if (weight_decay != 0.0f) {
                p *= wd_factor;
            }

            m = beta1 * m + (1.0f - beta1) * g;
            v = beta2 * v + (1.0f - beta2) * (g * g);
            if (!isfinite(v)) v = 0.0f;

            float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
            p -= step_size * (m / denom);

            uint32_t h = hash_coords(static_cast<uint32_t>(idx), step_seed, static_cast<uint32_t>(tensor_idx));
            param[idx] = from_float_sr<ParamT>(p, h);
            exp_avg[idx] = from_float_sr<StateM_T>(m, h ^ 0x55555555u);
            exp_avg_sq[idx] = from_float_sr<StateV_T>(v, h ^ 0xAAAAAAAAu);
        }
    }
    // Case 3: Fallback scalar path for unaligned tensors or mixed moment combinations
    else {
        for (int64_t idx = tid; idx < total_elements; idx += stride) {
            float p = to_float(param[idx]);
            float g = sanitize_grad_val(to_float(grad[idx]));
            float m = to_float(exp_avg[idx]);
            float v = to_float(exp_avg_sq[idx]);

            if (weight_decay != 0.0f) {
                p *= wd_factor;
            }

            m = beta1 * m + (1.0f - beta1) * g;
            v = beta2 * v + (1.0f - beta2) * (g * g);
            if (!isfinite(v)) v = 0.0f;

            float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
            p -= step_size * (m / denom);

            uint32_t h = hash_coords(static_cast<uint32_t>(idx), step_seed, static_cast<uint32_t>(tensor_idx));
            param[idx] = from_float_sr<ParamT>(p, h);
            exp_avg[idx] = from_float_sr<StateM_T>(m, h ^ 0x55555555u);
            exp_avg_sq[idx] = from_float_sr<StateV_T>(v, h ^ 0xAAAAAAAAu);
        }
    }
}

// ============================================================================
// Host Launcher: 2D Multi-Tensor Pointer-Array Launch
// ============================================================================

void launch_multi_tensor_adamw_cuda_2d(
    const std::vector<TensorMetadata>& host_metadata,
    int64_t max_numel,
    int param_dtype_code, // 0: float32, 1: bfloat16, 2: float16
    int state_m_dtype_code, // 0: float32, 1: bfloat16, 2: float16
    int state_v_dtype_code, // 0: float32, 1: bfloat16, 2: float16
    float lr,
    float beta1,
    float beta2,
    float eps,
    float step_size,
    float bias_correction2_sqrt,
    uint32_t step_seed,
    cudaStream_t stream
) {
    int num_tensors = static_cast<int>(host_metadata.size());
    if (num_tensors == 0 || max_numel <= 0) return;

    constexpr int MAX_BATCH_TENSORS = 512;
    for (int start = 0; start < num_tensors; start += MAX_BATCH_TENSORS) {
        int batch_size = std::min<int>(MAX_BATCH_TENSORS, num_tensors - start);

        // Copy batch metadata into constant memory asynchronously
        cudaMemcpyToSymbolAsync(
            c_tensor_metadata,
            host_metadata.data() + start,
            batch_size * sizeof(TensorMetadata),
            0,
            cudaMemcpyHostToDevice,
            stream
        );

        int threads = 256;
        int blocks_x = 1;
        for (int j = start; j < start + batch_size; ++j) {
            const TensorMetadata& meta = host_metadata[j];
            bool aligned16 =
                (reinterpret_cast<uintptr_t>(meta.param) % 16 == 0) &&
                (reinterpret_cast<uintptr_t>(meta.grad) % 16 == 0) &&
                (reinterpret_cast<uintptr_t>(meta.exp_avg) % 16 == 0) &&
                (reinterpret_cast<uintptr_t>(meta.exp_avg_sq) % 16 == 0);

            int vec_size = 1;
            if ((meta.flags & 1u) != 0u) {
                vec_size = 4;
            } else if (aligned16) {
                if (param_dtype_code == 0) {
                    vec_size = 4;
                } else if (param_dtype_code == state_m_dtype_code && param_dtype_code == state_v_dtype_code) {
                    vec_size = 8;
                }
            }
            int64_t work_items = (meta.numel + vec_size - 1) / vec_size;
            int tensor_blocks = static_cast<int>((work_items + threads - 1) / threads);
            blocks_x = std::max(blocks_x, tensor_blocks);
        }
        blocks_x = std::min(blocks_x, 512);

        // 2D Grid: grid.x processes tensor elements, grid.y selects the tensor
        dim3 grid(blocks_x, batch_size);
        dim3 block(threads);

        if (param_dtype_code == 1 && state_m_dtype_code == 1 && state_v_dtype_code == 1) {
            multi_tensor_adamw_kernel_2d<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 1 && state_m_dtype_code == 1 && state_v_dtype_code == 0) {
            multi_tensor_adamw_kernel_2d<__nv_bfloat16, __nv_bfloat16, float><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 2 && state_m_dtype_code == 2 && state_v_dtype_code == 2) {
            multi_tensor_adamw_kernel_2d<__half, __half, __half><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 2 && state_m_dtype_code == 2 && state_v_dtype_code == 0) {
            multi_tensor_adamw_kernel_2d<__half, __half, float><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 0 && state_m_dtype_code == 0 && state_v_dtype_code == 0) {
            multi_tensor_adamw_kernel_2d<float, float, float><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 0 && state_m_dtype_code == 1 && state_v_dtype_code == 1) {
            multi_tensor_adamw_kernel_2d<float, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 0 && state_m_dtype_code == 1 && state_v_dtype_code == 0) {
            multi_tensor_adamw_kernel_2d<float, __nv_bfloat16, float><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 0 && state_m_dtype_code == 2 && state_v_dtype_code == 2) {
            multi_tensor_adamw_kernel_2d<float, __half, __half><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 0 && state_m_dtype_code == 2 && state_v_dtype_code == 0) {
            multi_tensor_adamw_kernel_2d<float, __half, float><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 1 && state_m_dtype_code == 0 && state_v_dtype_code == 0) {
            multi_tensor_adamw_kernel_2d<__nv_bfloat16, float, float><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 2 && state_m_dtype_code == 0 && state_v_dtype_code == 0) {
            multi_tensor_adamw_kernel_2d<__half, float, float><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        }
    }
}

// ============================================================================
// Single Tensor Fused Kernel (Backwards Compatibility)
// ============================================================================

template<typename ParamT, typename StateM_T, typename StateV_T>
__global__ void single_tensor_adamw_kernel(
    ParamT* __restrict__ param,
    const ParamT* __restrict__ grad,
    StateM_T* __restrict__ exp_avg,
    StateV_T* __restrict__ exp_avg_sq,
    int64_t total_elements,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt,
    uint32_t step_seed
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const float wd_factor = (weight_decay != 0.0f) ? fmaxf(1.0f - lr * weight_decay, 0.0f) : 1.0f;

    constexpr bool is_pure_fp32 = std::is_same<ParamT, float>::value &&
                                  std::is_same<StateM_T, float>::value &&
                                  std::is_same<StateV_T, float>::value;
    if constexpr (is_pure_fp32) {
        for (; idx < total_elements; idx += stride) {
            float p = param[idx];
            float g = sanitize_grad_val(grad[idx]);
            float m = exp_avg[idx];
            float v = exp_avg_sq[idx];

            if (weight_decay != 0.0f) {
                p *= wd_factor;
            }

            m = beta1 * m + (1.0f - beta1) * g;
            v = beta2 * v + (1.0f - beta2) * (g * g);

            float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
            p -= step_size * (m / denom);

            param[idx] = p;
            exp_avg[idx] = m;
            exp_avg_sq[idx] = v;
        }
        return;
    }

    for (; idx < total_elements; idx += stride) {
        float p = to_float(param[idx]);
        float g = sanitize_grad_val(to_float(grad[idx]));
        float m = to_float(exp_avg[idx]);
        float v = to_float(exp_avg_sq[idx]);

        if (weight_decay != 0.0f) {
            p *= wd_factor;
        }

        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * (g * g);
        if (!isfinite(v)) v = 0.0f;

        float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
        p -= step_size * (m / denom);

        uint32_t h = hash_coords(static_cast<uint32_t>(idx), step_seed);
        param[idx] = from_float_sr<ParamT>(p, h);
        exp_avg[idx] = from_float_sr<StateM_T>(m, h ^ 0x55555555u);
        exp_avg_sq[idx] = from_float_sr<StateV_T>(v, h ^ 0xAAAAAAAAu);
    }
}

void launch_single_tensor_adamw_cuda(
    void* param,
    const void* grad,
    void* exp_avg,
    void* exp_avg_sq,
    int64_t total_elements,
    int param_dtype_code,
    int state_m_dtype_code,
    int state_v_dtype_code,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt,
    uint32_t step_seed,
    cudaStream_t stream
) {
    if (total_elements <= 0) return;

    int threads = 256;
    int64_t num_blocks = (total_elements + threads - 1) / threads;
    num_blocks = std::min<int64_t>(std::max<int64_t>(num_blocks, 1), 65535);

    if (param_dtype_code == 1 && state_m_dtype_code == 1 && state_v_dtype_code == 1) {
        single_tensor_adamw_kernel<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16><<<num_blocks, threads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(param),
            static_cast<const __nv_bfloat16*>(grad),
            static_cast<__nv_bfloat16*>(exp_avg),
            static_cast<__nv_bfloat16*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 1 && state_m_dtype_code == 1 && state_v_dtype_code == 0) {
        single_tensor_adamw_kernel<__nv_bfloat16, __nv_bfloat16, float><<<num_blocks, threads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(param),
            static_cast<const __nv_bfloat16*>(grad),
            static_cast<__nv_bfloat16*>(exp_avg),
            static_cast<float*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 2 && state_m_dtype_code == 2 && state_v_dtype_code == 2) {
        single_tensor_adamw_kernel<__half, __half, __half><<<num_blocks, threads, 0, stream>>>(
            static_cast<__half*>(param),
            static_cast<const __half*>(grad),
            static_cast<__half*>(exp_avg),
            static_cast<__half*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 2 && state_m_dtype_code == 2 && state_v_dtype_code == 0) {
        single_tensor_adamw_kernel<__half, __half, float><<<num_blocks, threads, 0, stream>>>(
            static_cast<__half*>(param),
            static_cast<const __half*>(grad),
            static_cast<__half*>(exp_avg),
            static_cast<float*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 0 && state_m_dtype_code == 0 && state_v_dtype_code == 0) {
        single_tensor_adamw_kernel<float, float, float><<<num_blocks, threads, 0, stream>>>(
            static_cast<float*>(param),
            static_cast<const float*>(grad),
            static_cast<float*>(exp_avg),
            static_cast<float*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 0 && state_m_dtype_code == 1 && state_v_dtype_code == 1) {
        single_tensor_adamw_kernel<float, __nv_bfloat16, __nv_bfloat16><<<num_blocks, threads, 0, stream>>>(
            static_cast<float*>(param),
            static_cast<const float*>(grad),
            static_cast<__nv_bfloat16*>(exp_avg),
            static_cast<__nv_bfloat16*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 0 && state_m_dtype_code == 1 && state_v_dtype_code == 0) {
        single_tensor_adamw_kernel<float, __nv_bfloat16, float><<<num_blocks, threads, 0, stream>>>(
            static_cast<float*>(param),
            static_cast<const float*>(grad),
            static_cast<__nv_bfloat16*>(exp_avg),
            static_cast<float*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 0 && state_m_dtype_code == 2 && state_v_dtype_code == 2) {
        single_tensor_adamw_kernel<float, __half, __half><<<num_blocks, threads, 0, stream>>>(
            static_cast<float*>(param),
            static_cast<const float*>(grad),
            static_cast<__half*>(exp_avg),
            static_cast<__half*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 0 && state_m_dtype_code == 2 && state_v_dtype_code == 0) {
        single_tensor_adamw_kernel<float, __half, float><<<num_blocks, threads, 0, stream>>>(
            static_cast<float*>(param),
            static_cast<const float*>(grad),
            static_cast<__half*>(exp_avg),
            static_cast<float*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 1 && state_m_dtype_code == 0 && state_v_dtype_code == 0) {
        single_tensor_adamw_kernel<__nv_bfloat16, float, float><<<num_blocks, threads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(param),
            static_cast<const __nv_bfloat16*>(grad),
            static_cast<float*>(exp_avg),
            static_cast<float*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 2 && state_m_dtype_code == 0 && state_v_dtype_code == 0) {
        single_tensor_adamw_kernel<__half, float, float><<<num_blocks, threads, 0, stream>>>(
            static_cast<__half*>(param),
            static_cast<const __half*>(grad),
            static_cast<float*>(exp_avg),
            static_cast<float*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    }
}

// ============================================================================
// 8-Bit (INT8 / NF4-INT4 Momentum + Log-UINT8 Variance) 2D Multi-Tensor Kernel
// ============================================================================

struct TensorMetadata8Bit {
    void* param;
    const void* grad;
    void* exp_avg;
    float* exp_avg_scales;
    uint8_t* exp_avg_sq;
    float* exp_avg_sq_scales;
    int64_t numel;
    int32_t block_size;
    uint8_t apply_wd;
    uint8_t pad[3];
};

// 576 * 64 = 36,864 bytes; combined with c_tensor_metadata (24,576) + c_nf4_lut (64) = 61,504 B < 64 KB limit
__constant__ TensorMetadata8Bit c_tensor_metadata_8bit[576];

__constant__ float c_nf4_lut[16] = {
    -1.0000f, -0.6962f, -0.5251f, -0.3949f,
    -0.2844f, -0.1848f, -0.0911f,  0.0000f,
     0.0911f,  0.1848f,  0.2844f,  0.3949f,
     0.5251f,  0.6962f,  1.0000f,  1.0000f
};

__device__ __forceinline__ uint8_t nf4_quantize_device(float norm_val, float rand_u, bool stochastic) {
    bool is_neg = norm_val < 0.0f;
    float ax = fabsf(norm_val);
    int lo;
    float c_lo, c_hi;
    if (ax < 0.2844f) {
        if (ax < 0.0911f) { lo = 0; c_lo = 0.0f; c_hi = 0.0911f; }
        else if (ax < 0.1848f) { lo = 1; c_lo = 0.0911f; c_hi = 0.1848f; }
        else { lo = 2; c_lo = 0.1848f; c_hi = 0.2844f; }
    } else if (ax < 0.5251f) {
        if (ax < 0.3949f) { lo = 3; c_lo = 0.2844f; c_hi = 0.3949f; }
        else { lo = 4; c_lo = 0.3949f; c_hi = 0.5251f; }
    } else {
        if (ax < 0.6962f) { lo = 5; c_lo = 0.5251f; c_hi = 0.6962f; }
        else { lo = 6; c_lo = 0.6962f; c_hi = 1.0000f; }
    }
    int k;
    if (stochastic) {
        float prob_hi = (ax - c_lo) / (c_hi - c_lo);
        k = (rand_u < prob_hi) ? (lo + 1) : lo;
    } else {
        k = ((ax - c_lo) >= (c_hi - ax)) ? (lo + 1) : lo;
    }
    return static_cast<uint8_t>(is_neg ? (7 - k) : (7 + k));
}

__device__ __forceinline__ void warp_block_reduce_max2(
    float& val1,
    float& val2,
    float* smem1,
    float* smem2
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val1 = fmaxf(val1, __shfl_xor_sync(0xffffffffu, val1, offset));
        val2 = fmaxf(val2, __shfl_xor_sync(0xffffffffu, val2, offset));
    }
    if (num_warps > 1) {
        if (lane == 0) {
            smem1[warp] = val1;
            smem2[warp] = val2;
        }
        __syncthreads();
        if (warp == 0) {
            val1 = (lane < num_warps) ? smem1[lane] : 0.0f;
            val2 = (lane < num_warps) ? smem2[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                val1 = fmaxf(val1, __shfl_xor_sync(0xffffffffu, val1, offset));
                val2 = fmaxf(val2, __shfl_xor_sync(0xffffffffu, val2, offset));
            }
            if (lane == 0) {
                smem1[0] = val1;
                smem2[0] = val2;
            }
        }
        __syncthreads();
        val1 = smem1[0];
        val2 = smem2[0];
        __syncthreads();
    } else {
        val1 = __shfl_sync(0xffffffffu, val1, 0);
        val2 = __shfl_sync(0xffffffffu, val2, 0);
    }
}

__device__ __forceinline__ float warp_block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    if (num_warps > 1) {
        if (lane == 0) smem[warp] = val;
        __syncthreads();
        if (warp == 0) {
            val = (lane < num_warps) ? smem[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                val += __shfl_xor_sync(0xffffffffu, val, offset);
            }
            if (lane == 0) smem[0] = val;
        }
        __syncthreads();
        val = smem[0];
        __syncthreads();
    } else {
        val = __shfl_sync(0xffffffffu, val, 0);
    }
    return val;
}

template<typename ParamT>
struct alignas(sizeof(ParamT) * 4) ApexParamVec4 {
    ParamT val[4];
};

struct alignas(4) ApexU8Vec4 {
    uint8_t val[4];
};

struct alignas(4) ApexI8Vec4 {
    int8_t val[4];
};

struct alignas(2) ApexU8Vec2 {
    uint8_t val[2];
};

template<typename ParamT, bool IS_4BIT_M, int ELEMS>
__global__ void multi_tensor_8bit_adamw_kernel_2d(
    int num_tensors,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float step_size,
    float eps_scaled,
    float inv_scale,
    uint32_t step_seed,
    bool stochastic,
    bool cautious
) {
    int tensor_idx = blockIdx.y;
    if (tensor_idx >= num_tensors) return;

    const TensorMetadata8Bit meta = c_tensor_metadata_8bit[tensor_idx];
    int64_t total_elements = meta.numel;
    if (total_elements <= 0) return;

    int32_t bs = meta.block_size;
    int64_t num_blocks = (total_elements + bs - 1) / bs;
    if (blockIdx.x >= num_blocks) return;

    __shared__ float smem_m[32];
    __shared__ float smem_v[32];

    ParamT* param = static_cast<ParamT*>(meta.param);
    const ParamT* grad = static_cast<const ParamT*>(meta.grad);
    float* m_scales = meta.exp_avg_scales;
    uint8_t* v_ptr = meta.exp_avg_sq;
    float* v_scales = meta.exp_avg_sq_scales;
    float wd_factor = meta.apply_wd ? fmaxf(1.0f - lr * weight_decay, 0.0f) : 1.0f;
    float one_minus_b1 = 1.0f - beta1;
    float one_minus_b2 = 1.0f - beta2;

    const bool is_quad_v = IS_4BIT_M && (meta.pad[0] != 0);
    const float gain_hi = 1.0f + 0.20f * beta1;
    const float gain_lo = 1.0f - 0.35f * beta1;
    const uintptr_t p_align_mask = static_cast<uintptr_t>(sizeof(ParamT) * 4 - 1);
    const bool ptrs_aligned4 = (ELEMS == 4) &&
        ((reinterpret_cast<uintptr_t>(param) & p_align_mask) == 0) &&
        ((reinterpret_cast<uintptr_t>(grad) & p_align_mask) == 0) &&
        (is_quad_v || ((reinterpret_cast<uintptr_t>(v_ptr) & 3u) == 0)) &&
        (IS_4BIT_M
            ? ((reinterpret_cast<uintptr_t>(meta.exp_avg) & 1u) == 0)
            : ((reinterpret_cast<uintptr_t>(meta.exp_avg) & 3u) == 0));

    for (int64_t b = blockIdx.x; b < num_blocks; b += gridDim.x) {
        int64_t block_start = b * bs;
        int64_t elem_base = block_start + static_cast<int64_t>(threadIdx.x) * ELEMS;
        bool full_vec4 = ptrs_aligned4 && ((block_start + bs) <= total_elements);

        float scale_m = m_scales[b];
        float scale_v = v_scales[b];

        float p_vals[ELEMS];
        float g_vals[ELEMS];
        float m_vals[ELEMS];
        float v_vals[ELEMS];
        bool valid[ELEMS];

        if constexpr (ELEMS == 4) {
            if (full_vec4) {
                const ApexParamVec4<ParamT> p_vec = *reinterpret_cast<const ApexParamVec4<ParamT>*>(param + elem_base);
                const ApexParamVec4<ParamT> g_vec = *reinterpret_cast<const ApexParamVec4<ParamT>*>(grad + elem_base);

                if (is_quad_v) {
                    uint8_t qv = v_ptr[elem_base >> 2];
                    float vq = (qv > 0) ? (exp2f((static_cast<float>(qv) - 1.0f) * (24.0f / 254.0f) - 24.0f) * scale_v) : 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        valid[k] = true;
                        p_vals[k] = to_float(p_vec.val[k]);
                        g_vals[k] = sanitize_grad_val(to_float(g_vec.val[k])) * inv_scale;
                        v_vals[k] = vq;
                    }
                } else {
                    const ApexU8Vec4 v_vec = *reinterpret_cast<const ApexU8Vec4*>(v_ptr + elem_base);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        valid[k] = true;
                        p_vals[k] = to_float(p_vec.val[k]);
                        g_vals[k] = sanitize_grad_val(to_float(g_vec.val[k])) * inv_scale;
                        uint8_t qv = v_vec.val[k];
                        v_vals[k] = (qv > 0) ? (exp2f((static_cast<float>(qv) - 1.0f) * (24.0f / 254.0f) - 24.0f) * scale_v) : 0.0f;
                    }
                }
                if constexpr (IS_4BIT_M) {
                    uint8_t* m_ptr = static_cast<uint8_t*>(meta.exp_avg);
                    const ApexU8Vec2 m_vec = *reinterpret_cast<const ApexU8Vec2*>(m_ptr + (elem_base >> 1));
                    m_vals[0] = c_nf4_lut[(m_vec.val[0] >> 4) & 0xF] * scale_m;
                    m_vals[1] = c_nf4_lut[m_vec.val[0] & 0xF] * scale_m;
                    m_vals[2] = c_nf4_lut[(m_vec.val[1] >> 4) & 0xF] * scale_m;
                    m_vals[3] = c_nf4_lut[m_vec.val[1] & 0xF] * scale_m;
                } else {
                    int8_t* m_ptr = static_cast<int8_t*>(meta.exp_avg);
                    const ApexI8Vec4 m_vec = *reinterpret_cast<const ApexI8Vec4*>(m_ptr + elem_base);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        m_vals[k] = static_cast<float>(m_vec.val[k]) * scale_m;
                    }
                }
            } else {
                if constexpr (IS_4BIT_M) {
                    uint8_t* m_ptr = static_cast<uint8_t*>(meta.exp_avg);
                    int64_t n_bytes = (total_elements + 1) >> 1;
                    int64_t byte0 = elem_base >> 1;
                    int64_t byte1 = byte0 + 1;
                    uint8_t pk0 = (byte0 < n_bytes) ? m_ptr[byte0] : 0x77;
                    uint8_t pk1 = (byte1 < n_bytes) ? m_ptr[byte1] : 0x77;
                    m_vals[0] = c_nf4_lut[(pk0 >> 4) & 0xF] * scale_m;
                    m_vals[1] = c_nf4_lut[pk0 & 0xF] * scale_m;
                    m_vals[2] = c_nf4_lut[(pk1 >> 4) & 0xF] * scale_m;
                    m_vals[3] = c_nf4_lut[pk1 & 0xF] * scale_m;
                } else {
                    int8_t* m_ptr = static_cast<int8_t*>(meta.exp_avg);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        int64_t idx = elem_base + k;
                        m_vals[k] = (idx < total_elements) ? (static_cast<float>(m_ptr[idx]) * scale_m) : 0.0f;
                    }
                }
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int64_t idx = elem_base + k;
                    bool ok = idx < total_elements;
                    valid[k] = ok;
                    p_vals[k] = ok ? to_float(param[idx]) : 0.0f;
                    g_vals[k] = ok ? sanitize_grad_val(to_float(grad[idx])) * inv_scale : 0.0f;
                    uint8_t qv = ok ? v_ptr[is_quad_v ? (idx >> 2) : idx] : 0;
                    v_vals[k] = (qv > 0) ? (exp2f((static_cast<float>(qv) - 1.0f) * (24.0f / 254.0f) - 24.0f) * scale_v) : 0.0f;
                }
            }
        } else {
            if constexpr (IS_4BIT_M) {
                uint8_t* m_ptr = static_cast<uint8_t*>(meta.exp_avg);
                int64_t n_bytes = (total_elements + 1) >> 1;
                #pragma unroll
                for (int p = 0; p < (ELEMS + 1) / 2; ++p) {
                    int64_t byte_idx = (elem_base >> 1) + p;
                    uint8_t packed = (byte_idx < n_bytes) ? m_ptr[byte_idx] : 0x77;
                    m_vals[p * 2] = c_nf4_lut[(packed >> 4) & 0xF] * scale_m;
                    if (p * 2 + 1 < ELEMS) {
                        m_vals[p * 2 + 1] = c_nf4_lut[packed & 0xF] * scale_m;
                    }
                }
            } else {
                int8_t* m_ptr = static_cast<int8_t*>(meta.exp_avg);
                #pragma unroll
                for (int k = 0; k < ELEMS; ++k) {
                    int64_t idx = elem_base + k;
                    m_vals[k] = (idx < total_elements) ? (static_cast<float>(m_ptr[idx]) * scale_m) : 0.0f;
                }
            }
            #pragma unroll
            for (int k = 0; k < ELEMS; ++k) {
                int64_t idx = elem_base + k;
                bool ok = idx < total_elements;
                valid[k] = ok;
                p_vals[k] = ok ? to_float(param[idx]) : 0.0f;
                g_vals[k] = ok ? sanitize_grad_val(to_float(grad[idx])) * inv_scale : 0.0f;
                uint8_t qv = ok ? v_ptr[idx] : 0;
                v_vals[k] = (qv > 0) ? (exp2f((static_cast<float>(qv) - 1.0f) * (24.0f / 254.0f) - 24.0f) * scale_v) : 0.0f;
            }
        }

        float u_vals[ELEMS];
        float local_max_m = 0.0f;
        float local_max_v = 0.0f;
        float local_n = 0.0f;
        float local_cm = 0.0f;

        if (is_quad_v && ELEMS == 4) {
            float g_sq_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < ELEMS; ++k) {
                if (meta.apply_wd) {
                    p_vals[k] *= wd_factor;
                }
                float g_k = g_vals[k];
                float m_k = m_vals[k] * beta1 + g_k * one_minus_b1;
                m_vals[k] = m_k;
                if (valid[k]) {
                    g_sq_sum += g_k * g_k;
                    local_max_m = fmaxf(local_max_m, fabsf(m_k));
                    if (cautious) {
                        local_n += 1.0f;
                        if (g_k * m_k >= 0.0f) {
                            local_cm += 1.0f;
                        }
                    }
                }
            }
            float v_q = v_vals[0] * beta2 + (0.25f * g_sq_sum) * one_minus_b2;
            if (!isfinite(v_q)) v_q = 0.0f;
            float true_eps = (eps > 1e-6f) ? 1e-8f : eps;
            float true_eps_scaled = eps_scaled * (true_eps / fmaxf(eps, 1e-12f));
            float inv_denom = step_size / fmaxf(sqrtf(fmaxf(v_q, 0.0f)) + true_eps_scaled, true_eps);
            if (valid[0]) {
                local_max_v = fmaxf(local_max_v, v_q);
            }
            #pragma unroll
            for (int k = 0; k < ELEMS; ++k) {
                v_vals[k] = v_q;
                float g_k = g_vals[k];
                float m_k = m_vals[k];
                float m_look = m_k * beta1 + g_k * one_minus_b1;
                float gain = (m_k * g_k >= 0.0f) ? gain_hi : gain_lo;
                u_vals[k] = (cautious ? m_k : (m_look * gain)) * inv_denom;
            }
        } else {
            #pragma unroll
            for (int k = 0; k < ELEMS; ++k) {
                if (meta.apply_wd) {
                    p_vals[k] *= wd_factor;
                }
                float g_k = g_vals[k];
                float m_k = m_vals[k] * beta1 + g_k * one_minus_b1;
                float v_k = v_vals[k] * beta2 + (g_k * g_k) * one_minus_b2;
                if (!isfinite(v_k)) v_k = 0.0f;
                m_vals[k] = m_k;
                v_vals[k] = v_k;
                float m_look = m_k * beta1 + g_k * one_minus_b1;
                float gain = (m_k * g_k >= 0.0f) ? gain_hi : gain_lo;
                float eff_m = cautious ? m_k : (m_look * gain);
                u_vals[k] = (eff_m / fmaxf(sqrtf(v_k) + eps_scaled, eps)) * step_size;
                if (valid[k]) {
                    local_max_m = fmaxf(local_max_m, fabsf(m_k));
                    local_max_v = fmaxf(local_max_v, v_k);
                    if (cautious) {
                        local_n += 1.0f;
                        if (g_k * m_k >= 0.0f) {
                            local_cm += 1.0f;
                        }
                    }
                }
            }
        }

        if (cautious) {
            float n_val = fmaxf(warp_block_reduce_sum(local_n, smem_m), 1.0f);
            float c_mean = fmaxf(warp_block_reduce_sum(local_cm, smem_v) / n_val, 1e-3f);
            #pragma unroll
            for (int k = 0; k < ELEMS; ++k) {
                float cm_k = (valid[k] && (g_vals[k] * m_vals[k] >= 0.0f)) ? 1.0f : 0.0f;
                u_vals[k] *= (cm_k / c_mean);
            }
        }

        // Single barrier pair reduces both max_m and max_v across the block
        warp_block_reduce_max2(local_max_m, local_max_v, smem_m, smem_v);

        float new_scale_v = fmaxf(fmaxf(local_max_v, scale_v * 0.95f), 1e-12f);
        float inv_sv = 1.0f / new_scale_v;

        uint32_t hashes[ELEMS];
        #pragma unroll
        for (int k = 0; k < ELEMS; ++k) {
            hashes[k] = stochastic ? hash_coords(static_cast<uint32_t>(elem_base + k), step_seed, static_cast<uint32_t>(tensor_idx)) : 0u;
        }

        if constexpr (ELEMS == 4) {
            if (full_vec4) {
                ApexParamVec4<ParamT> p_out;
                ApexU8Vec4 v_out;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    p_out.val[k] = from_float_sr<ParamT>(p_vals[k] - u_vals[k], hashes[k]);
                }
                *reinterpret_cast<ApexParamVec4<ParamT>*>(param + elem_base) = p_out;

                if constexpr (IS_4BIT_M) {
                    float new_scale_m = fmaxf(fmaxf(local_max_m, scale_m * 0.9f), 1e-12f);
                    float inv_sm = 1.0f / new_scale_m;
                    uint8_t nqs[4];
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        uint32_t hm = stochastic ? (hashes[k] * 1664525u + 1013904223u) : 0u;
                        float um = static_cast<float>(hm >> 16) * (1.0f / 65536.0f);
                        float nv = stochastic ? (static_cast<float>(hm & 0xffffu) * (1.0f / 65536.0f) - 0.5f) : 0.0f;
                        nqs[k] = nf4_quantize_device(fminf(fmaxf(m_vals[k] * inv_sm, -1.0f), 1.0f), um, stochastic);
                        if (!is_quad_v || k == 0) {
                            float vk = v_vals[k];
                            if (vk > 0.0f) {
                                float lv = log2f(fmaxf(vk * inv_sv, 5.9604645e-8f));
                                v_out.val[k] = static_cast<uint8_t>(fminf(fmaxf(floorf(lv * (254.0f / 24.0f) + (255.5f + nv)), 1.0f), 255.0f));
                            } else {
                                v_out.val[k] = 0;
                            }
                        }
                    }
                    ApexU8Vec2 m_out;
                    m_out.val[0] = static_cast<uint8_t>((nqs[0] << 4) | nqs[1]);
                    m_out.val[1] = static_cast<uint8_t>((nqs[2] << 4) | nqs[3]);
                    uint8_t* m_ptr = static_cast<uint8_t*>(meta.exp_avg);
                    *reinterpret_cast<ApexU8Vec2*>(m_ptr + (elem_base >> 1)) = m_out;
                    if (is_quad_v) {
                        v_ptr[elem_base >> 2] = v_out.val[0];
                    } else {
                        *reinterpret_cast<ApexU8Vec4*>(v_ptr + elem_base) = v_out;
                    }
                    if (threadIdx.x == 0) {
                        m_scales[b] = new_scale_m;
                        v_scales[b] = new_scale_v;
                    }
                } else {
                    float new_scale_m = fmaxf(fmaxf(local_max_m / 127.0f, scale_m * 0.9f), 1e-12f);
                    float inv_sm = 1.0f / new_scale_m;
                    ApexI8Vec4 m_out;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        uint32_t h = hashes[k];
                        float scaled_m = m_vals[k] * inv_sm;
                        float noise_v = 0.0f;
                        float rounded_m;
                        if (stochastic) {
                            float noise_m = static_cast<float>(h >> 16) * (1.0f / 65536.0f) - 0.5f;
                            noise_v = static_cast<float>(h & 0xffffu) * (1.0f / 65536.0f) - 0.5f;
                            rounded_m = floorf(scaled_m + noise_m + 0.5f);
                        } else {
                            rounded_m = (scaled_m >= 0.0f) ? floorf(scaled_m + 0.5f) : ceilf(scaled_m - 0.5f);
                        }
                        m_out.val[k] = static_cast<int8_t>(fminf(fmaxf(rounded_m, -127.0f), 127.0f));
                        float vk = v_vals[k];
                        if (vk > 0.0f) {
                            float lv = log2f(fmaxf(vk * inv_sv, 5.9604645e-8f));
                            v_out.val[k] = static_cast<uint8_t>(fminf(fmaxf(floorf(lv * (254.0f / 24.0f) + (255.5f + noise_v)), 1.0f), 255.0f));
                        } else {
                            v_out.val[k] = 0;
                        }
                    }
                    int8_t* m_ptr = static_cast<int8_t*>(meta.exp_avg);
                    *reinterpret_cast<ApexI8Vec4*>(m_ptr + elem_base) = m_out;
                    *reinterpret_cast<ApexU8Vec4*>(v_ptr + elem_base) = v_out;
                    if (threadIdx.x == 0) {
                        m_scales[b] = new_scale_m;
                        v_scales[b] = new_scale_v;
                    }
                }
                continue;
            }
        }

        // Tail block / unaligned view / non-4-element block fallback
        #pragma unroll
        for (int k = 0; k < ELEMS; ++k) {
            if (valid[k]) {
                param[elem_base + k] = from_float_sr<ParamT>(p_vals[k] - u_vals[k], hashes[k]);
            }
        }

        if constexpr (IS_4BIT_M) {
            float new_scale_m = fmaxf(fmaxf(local_max_m, scale_m * 0.9f), 1e-12f);
            float inv_sm = 1.0f / new_scale_m;
            uint8_t nqs[ELEMS];
            #pragma unroll
            for (int k = 0; k < ELEMS; ++k) {
                if (valid[k]) {
                    uint32_t hm = stochastic ? (hashes[k] * 1664525u + 1013904223u) : 0u;
                    float um = static_cast<float>(hm >> 16) * (1.0f / 65536.0f);
                    float nv = stochastic ? (static_cast<float>(hm & 0xffffu) * (1.0f / 65536.0f) - 0.5f) : 0.0f;
                    nqs[k] = nf4_quantize_device(fminf(fmaxf(m_vals[k] * inv_sm, -1.0f), 1.0f), um, stochastic);
                    if (!is_quad_v || k == 0) {
                        float vk = v_vals[k];
                        int64_t v_idx = is_quad_v ? ((elem_base + k) >> 2) : (elem_base + k);
                        if (vk > 0.0f) {
                            float lv = log2f(fmaxf(vk * inv_sv, 5.9604645e-8f));
                            v_ptr[v_idx] = static_cast<uint8_t>(fminf(fmaxf(floorf(lv * (254.0f / 24.0f) + (255.5f + nv)), 1.0f), 255.0f));
                        } else {
                            v_ptr[v_idx] = 0;
                        }
                    }
                } else {
                    nqs[k] = 7;
                }
            }
            uint8_t* m_ptr = static_cast<uint8_t*>(meta.exp_avg);
            int64_t n_bytes = (total_elements + 1) >> 1;
            #pragma unroll
            for (int p = 0; p < (ELEMS + 1) / 2; ++p) {
                int64_t byte_idx = (elem_base >> 1) + p;
                if (byte_idx < n_bytes) {
                    uint8_t hi = nqs[p * 2];
                    uint8_t lo = (p * 2 + 1 < ELEMS) ? nqs[p * 2 + 1] : 7;
                    m_ptr[byte_idx] = static_cast<uint8_t>((hi << 4) | lo);
                }
            }
            if (threadIdx.x == 0) {
                m_scales[b] = new_scale_m;
                v_scales[b] = new_scale_v;
            }
        } else {
            float new_scale_m = fmaxf(fmaxf(local_max_m / 127.0f, scale_m * 0.9f), 1e-12f);
            float inv_sm = 1.0f / new_scale_m;
            int8_t* m_ptr = static_cast<int8_t*>(meta.exp_avg);
            #pragma unroll
            for (int k = 0; k < ELEMS; ++k) {
                if (valid[k]) {
                    uint32_t h = hashes[k];
                    float scaled_m = m_vals[k] * inv_sm;
                    float noise_v = 0.0f;
                    float rounded_m;
                    if (stochastic) {
                        float noise_m = static_cast<float>(h >> 16) * (1.0f / 65536.0f) - 0.5f;
                        noise_v = static_cast<float>(h & 0xffffu) * (1.0f / 65536.0f) - 0.5f;
                        rounded_m = floorf(scaled_m + noise_m + 0.5f);
                    } else {
                        rounded_m = (scaled_m >= 0.0f) ? floorf(scaled_m + 0.5f) : ceilf(scaled_m - 0.5f);
                    }
                    m_ptr[elem_base + k] = static_cast<int8_t>(fminf(fmaxf(rounded_m, -127.0f), 127.0f));
                    float vk = v_vals[k];
                    if (vk > 0.0f) {
                        float lv = log2f(fmaxf(vk * inv_sv, 5.9604645e-8f));
                        v_ptr[elem_base + k] = static_cast<uint8_t>(fminf(fmaxf(floorf(lv * (254.0f / 24.0f) + (255.5f + noise_v)), 1.0f), 255.0f));
                    } else {
                        v_ptr[elem_base + k] = 0;
                    }
                }
            }
            if (threadIdx.x == 0) {
                m_scales[b] = new_scale_m;
                v_scales[b] = new_scale_v;
            }
        }
    }
}

template<typename ParamT, bool IS_4BIT_M>
static inline void dispatch_8bit_kernel_2d(
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    int elem_per_thread,
    int batch_size,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float step_size,
    float eps_scaled,
    float inv_scale,
    uint32_t step_seed,
    bool stochastic,
    bool cautious
) {
    if (elem_per_thread == 4) {
        multi_tensor_8bit_adamw_kernel_2d<ParamT, IS_4BIT_M, 4><<<grid, block, 0, stream>>>(
            batch_size, lr, beta1, beta2, eps, weight_decay, step_size, eps_scaled, inv_scale, step_seed, stochastic, cautious
        );
    } else if (elem_per_thread == 2) {
        multi_tensor_8bit_adamw_kernel_2d<ParamT, IS_4BIT_M, 2><<<grid, block, 0, stream>>>(
            batch_size, lr, beta1, beta2, eps, weight_decay, step_size, eps_scaled, inv_scale, step_seed, stochastic, cautious
        );
    } else {
        multi_tensor_8bit_adamw_kernel_2d<ParamT, IS_4BIT_M, 1><<<grid, block, 0, stream>>>(
            batch_size, lr, beta1, beta2, eps, weight_decay, step_size, eps_scaled, inv_scale, step_seed, stochastic, cautious
        );
    }
}

void launch_multi_tensor_8bit_adamw_cuda_2d(
    const std::vector<TensorMetadata8Bit>& host_metadata,
    int param_dtype_code,
    int block_size,
    bool is_4bit_m,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float step_size,
    float eps_scaled,
    float inv_scale,
    uint32_t step_seed,
    bool stochastic,
    bool cautious,
    cudaStream_t stream
) {
    int num_tensors = static_cast<int>(host_metadata.size());
    if (num_tensors == 0) return;

    int elem_per_thread = (block_size >= 128 && (block_size % 128) == 0)
        ? 4
        : ((block_size >= 64 && (block_size % 64) == 0) ? 2 : 1);
    int threads = std::max(32, block_size / elem_per_thread);
    constexpr int MAX_BATCH_8BIT = 576;

    for (int start = 0; start < num_tensors; start += MAX_BATCH_8BIT) {
        int batch_size = std::min<int>(MAX_BATCH_8BIT, num_tensors - start);
        cudaMemcpyToSymbolAsync(
            c_tensor_metadata_8bit,
            host_metadata.data() + start,
            batch_size * sizeof(TensorMetadata8Bit),
            0,
            cudaMemcpyHostToDevice,
            stream
        );

        int blocks_x = 1;
        for (int j = start; j < start + batch_size; ++j) {
            int tb = static_cast<int>((host_metadata[j].numel + block_size - 1) / block_size);
            blocks_x = std::max(blocks_x, tb);
        }
        blocks_x = std::min(blocks_x, 512);
        dim3 grid(blocks_x, batch_size);
        dim3 block(threads);

        if (is_4bit_m) {
            if (param_dtype_code == 1) {
                dispatch_8bit_kernel_2d<__nv_bfloat16, true>(grid, block, stream, elem_per_thread, batch_size, lr, beta1, beta2, eps, weight_decay, step_size, eps_scaled, inv_scale, step_seed, stochastic, cautious);
            } else if (param_dtype_code == 2) {
                dispatch_8bit_kernel_2d<__half, true>(grid, block, stream, elem_per_thread, batch_size, lr, beta1, beta2, eps, weight_decay, step_size, eps_scaled, inv_scale, step_seed, stochastic, cautious);
            } else if (param_dtype_code == 0) {
                dispatch_8bit_kernel_2d<float, true>(grid, block, stream, elem_per_thread, batch_size, lr, beta1, beta2, eps, weight_decay, step_size, eps_scaled, inv_scale, step_seed, stochastic, cautious);
            }
        } else {
            if (param_dtype_code == 1) {
                dispatch_8bit_kernel_2d<__nv_bfloat16, false>(grid, block, stream, elem_per_thread, batch_size, lr, beta1, beta2, eps, weight_decay, step_size, eps_scaled, inv_scale, step_seed, stochastic, cautious);
            } else if (param_dtype_code == 2) {
                dispatch_8bit_kernel_2d<__half, false>(grid, block, stream, elem_per_thread, batch_size, lr, beta1, beta2, eps, weight_decay, step_size, eps_scaled, inv_scale, step_seed, stochastic, cautious);
            } else if (param_dtype_code == 0) {
                dispatch_8bit_kernel_2d<float, false>(grid, block, stream, elem_per_thread, batch_size, lr, beta1, beta2, eps, weight_decay, step_size, eps_scaled, inv_scale, step_seed, stochastic, cautious);
            }
        }
    }
}
