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

__device__ __forceinline__ uint32_t hash_coords(uint32_t idx, uint32_t step) {
    uint32_t h = idx ^ (step * 0x9E3779B9u);
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
    mag += (rand_val & 0xFFFFu);
    u = (u & 0x80000000u) | (mag & 0x7FFFFFFFu);
    uint16_t upper = static_cast<uint16_t>(u >> 16);
    return *reinterpret_cast<__nv_bfloat16*>(&upper);
}

__device__ __forceinline__ __half float_to_f16_sr(float x, uint32_t rand_val) {
    uint32_t u = __float_as_uint(x);
    uint32_t mag = u & 0x7FFFFFFFu;
    mag += (rand_val & 0x1FFFu); // 13 bits of discarded mantissa in FP16
    u = (u & 0x80000000u) | (mag & 0x7FFFFFFFu);
    float rounded = __uint_as_float(u);
    return __float2half_rz(rounded);
}

// ============================================================================
// Type Conversion & Arithmetic Helpers
// ============================================================================

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

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
};

// 64 KB constant memory holds up to 1024 tensors (~48 KB)
__constant__ TensorMetadata c_tensor_metadata[1024];

// ============================================================================
// 2D Grid Multi-Tensor Pointer-Array Fused Kernel with Stochastic Rounding
//
// Grid Configuration:
// - blockIdx.y selects the tensor index in the batch (0 <= blockIdx.y < num_tensors)
// - blockIdx.x * blockDim.x + threadIdx.x determines elements inside that tensor
// ============================================================================

template<typename ParamT, typename StateT>
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
    ParamT* param = static_cast<ParamT*>(meta.param);
    const ParamT* grad = static_cast<const ParamT*>(meta.grad);
    StateT* exp_avg = static_cast<StateT*>(meta.exp_avg);
    StateT* exp_avg_sq = static_cast<StateT*>(meta.exp_avg_sq);

    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    bool is_aligned16 = (((uintptr_t)param % 16) == 0) &&
                        (((uintptr_t)grad % 16) == 0) &&
                        (((uintptr_t)exp_avg % 16) == 0) &&
                        (((uintptr_t)exp_avg_sq % 16) == 0);

    constexpr int VEC_PARAM = 16 / sizeof(ParamT);

    // Case 1: Both param and state have matching element size (e.g. BF16/BF16, FP16/FP16, FP32/FP32)
    if (is_aligned16 && (sizeof(ParamT) == sizeof(StateT))) {
        int64_t vec_count = total_elements / VEC_PARAM;
        for (int64_t idx = tid; idx < vec_count; idx += stride) {
            float4 p_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(param) + idx * 16);
            float4 g_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(grad) + idx * 16);
            float4 m_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(exp_avg) + idx * 16);
            float4 v_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(exp_avg_sq) + idx * 16);

            ParamT* p_arr = reinterpret_cast<ParamT*>(&p_raw);
            const ParamT* g_arr = reinterpret_cast<const ParamT*>(&g_raw);
            StateT* m_arr = reinterpret_cast<StateT*>(&m_raw);
            StateT* v_arr = reinterpret_cast<StateT*>(&v_raw);

            uint32_t base_h = hash_coords(static_cast<uint32_t>(idx * VEC_PARAM), step_seed);

            #pragma unroll
            for (int i = 0; i < VEC_PARAM; ++i) {
                float p = to_float(p_arr[i]);
                float g = to_float(g_arr[i]);
                float m = to_float(m_arr[i]);
                float v = to_float(v_arr[i]);

                if (weight_decay != 0.0f) {
                    p -= lr * weight_decay * p;
                }

                m = beta1 * m + (1.0f - beta1) * g;
                v = beta2 * v + (1.0f - beta2) * (g * g);

                // Adaptive dynamic epsilon: full FP32 promotion and division-by-zero protection
                float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
                p -= step_size * (m / denom);

                uint32_t hi = base_h + i * 0x9E3779B9u;
                p_arr[i] = from_float_sr<ParamT>(p, hi);
                m_arr[i] = from_float_sr<StateT>(m, hi ^ 0x55555555u);
                v_arr[i] = from_float_sr<StateT>(v, hi ^ 0xAAAAAAAAu);
            }

            *reinterpret_cast<float4*>(reinterpret_cast<char*>(param) + idx * 16) = p_raw;
            *reinterpret_cast<float4*>(reinterpret_cast<char*>(exp_avg) + idx * 16) = m_raw;
            *reinterpret_cast<float4*>(reinterpret_cast<char*>(exp_avg_sq) + idx * 16) = v_raw;
        }

        // Scalar tail loop
        int64_t vec_limit = vec_count * VEC_PARAM;
        for (int64_t idx = vec_limit + tid; idx < total_elements; idx += stride) {
            float p = to_float(param[idx]);
            float g = to_float(grad[idx]);
            float m = to_float(exp_avg[idx]);
            float v = to_float(exp_avg_sq[idx]);

            if (weight_decay != 0.0f) {
                p -= lr * weight_decay * p;
            }

            m = beta1 * m + (1.0f - beta1) * g;
            v = beta2 * v + (1.0f - beta2) * (g * g);

            float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
            p -= step_size * (m / denom);

            uint32_t h = hash_coords(static_cast<uint32_t>(idx), step_seed);
            param[idx] = from_float_sr<ParamT>(p, h);
            exp_avg[idx] = from_float_sr<StateT>(m, h ^ 0x55555555u);
            exp_avg_sq[idx] = from_float_sr<StateT>(v, h ^ 0xAAAAAAAAu);
        }
    }
    // Case 2: 32-bit params + 16-bit compact states (FP32 weights with BF16/FP16 moments)
    else if (is_aligned16 && (sizeof(ParamT) == 4) && (sizeof(StateT) == 2)) {
        int64_t vec_count = total_elements / 4;
        for (int64_t idx = tid; idx < vec_count; idx += stride) {
            float4 p_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(param) + idx * 16);
            float4 g_raw = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(grad) + idx * 16);
            uint2 m_raw = *reinterpret_cast<const uint2*>(reinterpret_cast<const char*>(exp_avg) + idx * 8);
            uint2 v_raw = *reinterpret_cast<const uint2*>(reinterpret_cast<const char*>(exp_avg_sq) + idx * 8);

            float* p_arr = reinterpret_cast<float*>(&p_raw);
            const float* g_arr = reinterpret_cast<const float*>(&g_raw);
            StateT* m_arr = reinterpret_cast<StateT*>(&m_raw);
            StateT* v_arr = reinterpret_cast<StateT*>(&v_raw);

            uint32_t base_h = hash_coords(static_cast<uint32_t>(idx * 4), step_seed);

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float p = p_arr[i];
                float g = g_arr[i];
                float m = to_float(m_arr[i]);
                float v = to_float(v_arr[i]);

                if (weight_decay != 0.0f) {
                    p -= lr * weight_decay * p;
                }

                m = beta1 * m + (1.0f - beta1) * g;
                v = beta2 * v + (1.0f - beta2) * (g * g);

                float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
                p -= step_size * (m / denom);

                uint32_t hi = base_h + i * 0x9E3779B9u;
                p_arr[i] = p; // Float32 param requires no downcasting
                m_arr[i] = from_float_sr<StateT>(m, hi ^ 0x55555555u);
                v_arr[i] = from_float_sr<StateT>(v, hi ^ 0xAAAAAAAAu);
            }

            *reinterpret_cast<float4*>(reinterpret_cast<char*>(param) + idx * 16) = p_raw;
            *reinterpret_cast<uint2*>(reinterpret_cast<char*>(exp_avg) + idx * 8) = m_raw;
            *reinterpret_cast<uint2*>(reinterpret_cast<char*>(exp_avg_sq) + idx * 8) = v_raw;
        }

        // Scalar tail loop
        int64_t vec_limit = vec_count * 4;
        for (int64_t idx = vec_limit + tid; idx < total_elements; idx += stride) {
            float p = to_float(param[idx]);
            float g = to_float(grad[idx]);
            float m = to_float(exp_avg[idx]);
            float v = to_float(exp_avg_sq[idx]);

            if (weight_decay != 0.0f) {
                p -= lr * weight_decay * p;
            }

            m = beta1 * m + (1.0f - beta1) * g;
            v = beta2 * v + (1.0f - beta2) * (g * g);

            float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
            p -= step_size * (m / denom);

            uint32_t h = hash_coords(static_cast<uint32_t>(idx), step_seed);
            param[idx] = from_float_sr<ParamT>(p, h);
            exp_avg[idx] = from_float_sr<StateT>(m, h ^ 0x55555555u);
            exp_avg_sq[idx] = from_float_sr<StateT>(v, h ^ 0xAAAAAAAAu);
        }
    }
    // Case 3: Fallback scalar path for unaligned tensors or other combinations
    else {
        for (int64_t idx = tid; idx < total_elements; idx += stride) {
            float p = to_float(param[idx]);
            float g = to_float(grad[idx]);
            float m = to_float(exp_avg[idx]);
            float v = to_float(exp_avg_sq[idx]);

            if (weight_decay != 0.0f) {
                p -= lr * weight_decay * p;
            }

            m = beta1 * m + (1.0f - beta1) * g;
            v = beta2 * v + (1.0f - beta2) * (g * g);

            float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
            p -= step_size * (m / denom);

            uint32_t h = hash_coords(static_cast<uint32_t>(idx), step_seed);
            param[idx] = from_float_sr<ParamT>(p, h);
            exp_avg[idx] = from_float_sr<StateT>(m, h ^ 0x55555555u);
            exp_avg_sq[idx] = from_float_sr<StateT>(v, h ^ 0xAAAAAAAAu);
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
    int state_dtype_code, // 0: float32, 1: bfloat16, 2: float16
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

    constexpr int MAX_BATCH_TENSORS = 1024;
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
        int vec_size = (param_dtype_code == 0 && state_dtype_code != 0) ? 4 : (param_dtype_code == 1 ? 8 : 4);
        int64_t max_vec = (max_numel + vec_size - 1) / vec_size;
        int blocks_x = static_cast<int>((max_vec + threads - 1) / threads);
        blocks_x = std::max(1, std::min(blocks_x, 512));

        // 2D Grid: grid.x processes tensor elements, grid.y selects the tensor
        dim3 grid(blocks_x, batch_size);
        dim3 block(threads);

        if (param_dtype_code == 1 && state_dtype_code == 1) {
            multi_tensor_adamw_kernel_2d<__nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 0 && state_dtype_code == 1) {
            multi_tensor_adamw_kernel_2d<float, __nv_bfloat16><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 2 && state_dtype_code == 2) {
            multi_tensor_adamw_kernel_2d<__half, __half><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 0 && state_dtype_code == 2) {
            multi_tensor_adamw_kernel_2d<float, __half><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 0 && state_dtype_code == 0) {
            multi_tensor_adamw_kernel_2d<float, float><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 1 && state_dtype_code == 0) {
            multi_tensor_adamw_kernel_2d<__nv_bfloat16, float><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        } else if (param_dtype_code == 2 && state_dtype_code == 0) {
            multi_tensor_adamw_kernel_2d<__half, float><<<grid, block, 0, stream>>>(
                batch_size, lr, beta1, beta2, eps, step_size, bias_correction2_sqrt, step_seed
            );
        }
    }
}

// ============================================================================
// Single Tensor Fused Kernel (Backwards Compatibility)
// ============================================================================

template<typename ParamT, typename StateT>
__global__ void single_tensor_adamw_kernel(
    ParamT* __restrict__ param,
    const ParamT* __restrict__ grad,
    StateT* __restrict__ exp_avg,
    StateT* __restrict__ exp_avg_sq,
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

    for (; idx < total_elements; idx += stride) {
        float p = to_float(param[idx]);
        float g = to_float(grad[idx]);
        float m = to_float(exp_avg[idx]);
        float v = to_float(exp_avg_sq[idx]);

        if (weight_decay != 0.0f) {
            p -= lr * weight_decay * p;
        }

        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * (g * g);

        float denom = fmaxf((sqrtf(fmaxf(v, 0.0f)) / bias_correction2_sqrt) + eps, eps);
        p -= step_size * (m / denom);

        uint32_t h = hash_coords(static_cast<uint32_t>(idx), step_seed);
        param[idx] = from_float_sr<ParamT>(p, h);
        exp_avg[idx] = from_float_sr<StateT>(m, h ^ 0x55555555u);
        exp_avg_sq[idx] = from_float_sr<StateT>(v, h ^ 0xAAAAAAAAu);
    }
}

void launch_single_tensor_adamw_cuda(
    void* param,
    const void* grad,
    void* exp_avg,
    void* exp_avg_sq,
    int64_t total_elements,
    int param_dtype_code,
    int state_dtype_code,
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

    if (param_dtype_code == 1 && state_dtype_code == 1) {
        single_tensor_adamw_kernel<__nv_bfloat16, __nv_bfloat16><<<num_blocks, threads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(param),
            static_cast<const __nv_bfloat16*>(grad),
            static_cast<__nv_bfloat16*>(exp_avg),
            static_cast<__nv_bfloat16*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 0 && state_dtype_code == 1) {
        single_tensor_adamw_kernel<float, __nv_bfloat16><<<num_blocks, threads, 0, stream>>>(
            static_cast<float*>(param),
            static_cast<const float*>(grad),
            static_cast<__nv_bfloat16*>(exp_avg),
            static_cast<__nv_bfloat16*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 2 && state_dtype_code == 2) {
        single_tensor_adamw_kernel<__half, __half><<<num_blocks, threads, 0, stream>>>(
            static_cast<__half*>(param),
            static_cast<const __half*>(grad),
            static_cast<__half*>(exp_avg),
            static_cast<__half*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 0 && state_dtype_code == 2) {
        single_tensor_adamw_kernel<float, __half><<<num_blocks, threads, 0, stream>>>(
            static_cast<float*>(param),
            static_cast<const float*>(grad),
            static_cast<__half*>(exp_avg),
            static_cast<__half*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 0 && state_dtype_code == 0) {
        single_tensor_adamw_kernel<float, float><<<num_blocks, threads, 0, stream>>>(
            static_cast<float*>(param),
            static_cast<const float*>(grad),
            static_cast<float*>(exp_avg),
            static_cast<float*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 1 && state_dtype_code == 0) {
        single_tensor_adamw_kernel<__nv_bfloat16, float><<<num_blocks, threads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(param),
            static_cast<const __nv_bfloat16*>(grad),
            static_cast<float*>(exp_avg),
            static_cast<float*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    } else if (param_dtype_code == 2 && state_dtype_code == 0) {
        single_tensor_adamw_kernel<__half, float><<<num_blocks, threads, 0, stream>>>(
            static_cast<__half*>(param),
            static_cast<const __half*>(grad),
            static_cast<float*>(exp_avg),
            static_cast<float*>(exp_avg_sq),
            total_elements, lr, beta1, beta2, weight_decay, eps, step_size, bias_correction2_sqrt, step_seed
        );
    }
}
