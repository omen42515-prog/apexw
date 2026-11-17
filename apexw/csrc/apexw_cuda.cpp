#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_map>

struct TensorMetadata {
    void* param;
    const void* grad;
    void* exp_avg;
    void* exp_avg_sq;
    int64_t numel;
    float weight_decay;
    uint32_t flags;
};

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

// Forward declarations from apexw_cuda_kernel.cu
void launch_multi_tensor_adamw_cuda_2d(
    const std::vector<TensorMetadata>& host_metadata,
    int64_t max_numel,
    int param_dtype_code,
    int state_dtype_code,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float step_size,
    float bias_correction2_sqrt,
    uint32_t step_seed,
    cudaStream_t stream
);

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
);

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
);

static inline int get_dtype_code(at::ScalarType dtype) {
    if (dtype == at::ScalarType::Float) return 0;
    if (dtype == at::ScalarType::BFloat16) return 1;
    if (dtype == at::ScalarType::Half) return 2;
    return -1;
}

static inline bool is_supported_dtype_pair(int param_code, int state_code) {
    return !(param_code == 1 && state_code == 2) && !(param_code == 2 && state_code == 1);
}

void apexw_cuda_step_fused(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step,
    bool decay_norm_only
) {
    const size_t num_tensors = params.size();
    if (num_tensors == 0) return;

    TORCH_CHECK(grads.size() == num_tensors, "grads list length mismatch");
    TORCH_CHECK(exp_avgs.size() == num_tensors, "exp_avgs list length mismatch");
    TORCH_CHECK(exp_avg_sqs.size() == num_tensors, "exp_avg_sqs list length mismatch");
    TORCH_CHECK(params[0].is_cuda(), "All tensors must be CUDA tensors");

    const c10::cuda::CUDAGuard device_guard(params[0].device());
    const float f_lr = static_cast<float>(lr);
    const float f_beta1 = static_cast<float>(beta1);
    const float f_beta2 = static_cast<float>(beta2);
    const float f_eps = static_cast<float>(eps);
    const float f_wd = static_cast<float>(weight_decay);

    // Compute bias corrections in double precision before narrowing to float
    const double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    const double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));
    const float step_size = static_cast<float>(lr / bias_correction1);
    const float bias_correction2_sqrt = static_cast<float>(std::sqrt(bias_correction2));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(params[0].get_device()).stream();

    std::unordered_map<int, std::vector<size_t>> dtype_groups;
    for (size_t i = 0; i < num_tensors; ++i) {
        torch::Tensor& p = params[i];
        const torch::Tensor& g = grads[i];
        torch::Tensor& m = exp_avgs[i];
        torch::Tensor& v = exp_avg_sqs[i];

        TORCH_CHECK(p.is_cuda() && g.is_cuda() && m.is_cuda() && v.is_cuda(),
                    "All tensors must be CUDA tensors");
        TORCH_CHECK(p.device() == params[0].device() && g.device() == p.device() &&
                    m.device() == p.device() && v.device() == p.device(),
                    "All tensors must be on the same CUDA device");
        TORCH_CHECK(p.is_contiguous() && g.is_contiguous() && m.is_contiguous() && v.is_contiguous(),
                    "All tensors must be contiguous");
        bool is_quad_v = (p.numel() >= 4) && (p.numel() % 4 == 0) && (v.numel() * 4 == p.numel());
        TORCH_CHECK(p.numel() == g.numel() && p.numel() == m.numel() && (p.numel() == v.numel() || is_quad_v),
                    "All tensors for a parameter must have matching element counts");
        TORCH_CHECK(g.scalar_type() == p.scalar_type(),
                    "grad and param must have matching dtype");

        int param_code = get_dtype_code(p.scalar_type());
        int state_code = get_dtype_code(m.scalar_type());

        TORCH_CHECK(param_code != -1, "ApexW CUDA supports Float32, BFloat16, and Float16 parameters");
        TORCH_CHECK(state_code != -1, "ApexW CUDA supports Float32, BFloat16, and Float16 optimizer states");
        TORCH_CHECK(is_supported_dtype_pair(param_code, state_code),
                    "Cross-16-bit (BF16 param with FP16 state or vice versa) is not supported by ApexW CUDA kernel");
        TORCH_CHECK(m.scalar_type() == v.scalar_type(),
                    "ApexW CUDA kernel requires exp_avg and exp_avg_sq to share the same dtype.");

        int key = ((is_quad_v ? 1 : 0) << 8) | (param_code << 4) | state_code;
        dtype_groups[key].push_back(i);
    }

    for (const auto& kv : dtype_groups) {
        int key = kv.first;
        bool is_quad_v = ((key >> 8) & 1) != 0;
        int param_code = (key >> 4) & 0xF;
        int state_code = key & 0xF;
        const auto& indices = kv.second;

        std::vector<TensorMetadata> host_metadata(indices.size());
        int64_t max_numel = 0;

        for (size_t j = 0; j < indices.size(); ++j) {
            size_t idx = indices[j];
            torch::Tensor& p = params[idx];
            const torch::Tensor& g = grads[idx];
            torch::Tensor& m = exp_avgs[idx];
            torch::Tensor& v = exp_avg_sqs[idx];

            int64_t numel = p.numel();
            if (numel > max_numel) {
                max_numel = numel;
            }

            host_metadata[j].param = p.data_ptr();
            host_metadata[j].grad = g.data_ptr();
            host_metadata[j].exp_avg = m.data_ptr();
            host_metadata[j].exp_avg_sq = v.data_ptr();
            host_metadata[j].numel = numel;

            float wd = (decay_norm_only && p.dim() < 2) ? 0.0f : f_wd;
            host_metadata[j].weight_decay = wd;
            host_metadata[j].flags = is_quad_v ? 1u : 0u;
        }

        if (max_numel > 0) {
            const float bucket_eps = (param_code == 2 || state_code == 2) ? std::max(f_eps, 6.1e-5f) : f_eps;
            launch_multi_tensor_adamw_cuda_2d(
                host_metadata,
                max_numel,
                param_code,
                state_code,
                f_lr,
                f_beta1,
                f_beta2,
                bucket_eps,
                step_size,
                bias_correction2_sqrt,
                static_cast<uint32_t>(step),
                stream
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
}

void apexw_fused_adamw_cuda_step(
    torch::Tensor& param,
    const torch::Tensor& grad,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps,
    int64_t step
) {
    TORCH_CHECK(param.is_cuda() && grad.is_cuda() && exp_avg.is_cuda() && exp_avg_sq.is_cuda(),
                "All tensors must be CUDA tensors");
    const c10::cuda::CUDAGuard device_guard(param.device());
    TORCH_CHECK(grad.device() == param.device() && exp_avg.device() == param.device() && exp_avg_sq.device() == param.device(),
                "All tensors must be on the same CUDA device");
    TORCH_CHECK(param.is_contiguous() && grad.is_contiguous() && exp_avg.is_contiguous() && exp_avg_sq.is_contiguous(),
                "All tensors must be contiguous");
    TORCH_CHECK(param.numel() == grad.numel() && param.numel() == exp_avg.numel() && param.numel() == exp_avg_sq.numel(),
                "All tensors must have matching element counts");
    TORCH_CHECK(grad.scalar_type() == param.scalar_type(), "grad and param must have matching dtype");

    int param_code = get_dtype_code(param.scalar_type());
    int state_code = get_dtype_code(exp_avg.scalar_type());

    TORCH_CHECK(param_code != -1, "ApexW CUDA supports Float32, BFloat16, and Float16 parameters");
    TORCH_CHECK(state_code != -1, "ApexW CUDA supports Float32, BFloat16, and Float16 optimizer states");
    TORCH_CHECK(is_supported_dtype_pair(param_code, state_code),
                "Cross-16-bit (BF16 param with FP16 state or vice versa) is not supported by ApexW CUDA kernel");
    TORCH_CHECK(exp_avg.scalar_type() == exp_avg_sq.scalar_type(), "exp_avg and exp_avg_sq must share same dtype");

    const int64_t total_elements = param.numel();
    if (total_elements == 0) return;

    const float f_lr = static_cast<float>(lr);
    const float f_beta1 = static_cast<float>(beta1);
    const float f_beta2 = static_cast<float>(beta2);
    const float f_wd = static_cast<float>(weight_decay);
    const float f_eps = static_cast<float>(eps);

    const double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    const double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));
    const float step_size = static_cast<float>(lr / bias_correction1);
    const float bias_correction2_sqrt = static_cast<float>(std::sqrt(bias_correction2));

    const float eff_eps = (param_code == 2 || state_code == 2) ? std::max(f_eps, 6.1e-5f) : f_eps;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(param.get_device()).stream();

    launch_single_tensor_adamw_cuda(
        param.data_ptr(),
        grad.data_ptr(),
        exp_avg.data_ptr(),
        exp_avg_sq.data_ptr(),
        total_elements,
        param_code,
        state_code,
        f_lr,
        f_beta1,
        f_beta2,
        f_wd,
        eff_eps,
        step_size,
        bias_correction2_sqrt,
        static_cast<uint32_t>(step),
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void apexw_cuda_step_fused_8bit(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_scales,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& exp_avg_sq_scales,
    const std::vector<int64_t>& block_sizes,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step,
    bool decay_norm_only,
    double inv_scale,
    bool stochastic,
    bool cautious
) {
    const size_t num_tensors = params.size();
    if (num_tensors == 0) return;

    TORCH_CHECK(grads.size() == num_tensors, "grads list length mismatch");
    TORCH_CHECK(exp_avgs.size() == num_tensors, "exp_avgs list length mismatch");
    TORCH_CHECK(exp_avg_scales.size() == num_tensors, "exp_avg_scales list length mismatch");
    TORCH_CHECK(exp_avg_sqs.size() == num_tensors, "exp_avg_sqs list length mismatch");
    TORCH_CHECK(exp_avg_sq_scales.size() == num_tensors, "exp_avg_sq_scales list length mismatch");
    TORCH_CHECK(block_sizes.size() == num_tensors, "block_sizes list length mismatch");
    TORCH_CHECK(params[0].is_cuda(), "All tensors must be CUDA tensors");

    const c10::cuda::CUDAGuard device_guard(params[0].device());
    const float f_lr = static_cast<float>(lr);
    const float f_beta1 = static_cast<float>(beta1);
    const float f_beta2 = static_cast<float>(beta2);
    const float f_eps = static_cast<float>(eps);
    const float f_wd = static_cast<float>(weight_decay);
    const float f_inv_scale = static_cast<float>(inv_scale);

    const double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    const double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));
    const double bc2_sqrt = std::sqrt(bias_correction2);
    const float step_size = static_cast<float>(lr * bc2_sqrt / bias_correction1);
    const float eps_scaled = static_cast<float>(eps * bc2_sqrt);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(params[0].get_device()).stream();

    // Group by (param_code, is_4bit_m, block_size)
    std::unordered_map<int, std::vector<size_t>> buckets;
    for (size_t i = 0; i < num_tensors; ++i) {
        torch::Tensor& p = params[i];
        const torch::Tensor& g = grads[i];
        torch::Tensor& m = exp_avgs[i];
        torch::Tensor& ms = exp_avg_scales[i];
        torch::Tensor& v = exp_avg_sqs[i];
        torch::Tensor& vs = exp_avg_sq_scales[i];
        int64_t bs = block_sizes[i];

        TORCH_CHECK(p.is_cuda() && g.is_cuda() && m.is_cuda() && ms.is_cuda() && v.is_cuda() && vs.is_cuda(),
                    "All tensors must be CUDA tensors");
        TORCH_CHECK(p.device() == params[0].device() && g.device() == p.device() &&
                    m.device() == p.device() && ms.device() == p.device() &&
                    v.device() == p.device() && vs.device() == p.device(),
                    "All tensors must be on the same CUDA device");
        TORCH_CHECK(p.is_contiguous() && g.is_contiguous() && m.is_contiguous() &&
                    ms.is_contiguous() && v.is_contiguous() && vs.is_contiguous(),
                    "All tensors must be contiguous");
        TORCH_CHECK(p.scalar_type() == g.scalar_type(), "grad and param must have matching dtype");
        TORCH_CHECK(ms.scalar_type() == at::ScalarType::Float && vs.scalar_type() == at::ScalarType::Float,
                    "Scales must be float32");
        TORCH_CHECK(v.scalar_type() == at::ScalarType::Byte, "exp_avg_sq must be uint8");
        TORCH_CHECK(bs == 64 || bs == 128 || bs == 256 || bs == 512 || bs == 1024 || bs == 2048,
                    "CUDA 8bit supports block_size 64, 128, 256, 512, 1024, or 2048");

        int param_code = get_dtype_code(p.scalar_type());
        TORCH_CHECK(param_code != -1, "Unsupported param dtype for CUDA 8bit");

        bool is_4bit_m = (m.scalar_type() == at::ScalarType::Byte);
        bool is_quad_v = is_4bit_m && (p.numel() >= 4) && (p.numel() % 4 == 0) && (v.numel() * 4 == p.numel());
        TORCH_CHECK(is_4bit_m || m.scalar_type() == at::ScalarType::Char, "exp_avg must be int8 or packed uint8");
        TORCH_CHECK(p.numel() == g.numel() && (p.numel() == v.numel() || is_quad_v), "Element count mismatch");
        if (is_4bit_m) {
            TORCH_CHECK(m.numel() == (p.numel() + 1) / 2, "NF4 packed exp_avg size mismatch");
        } else {
            TORCH_CHECK(m.numel() == p.numel(), "INT8 exp_avg size mismatch");
        }

        if (p.numel() == 0) continue;
        int key = (param_code << 16) | ((is_4bit_m ? 1 : 0) << 15) | ((is_quad_v ? 1 : 0) << 14) | (static_cast<int>(bs) & 0x3FFF);
        buckets[key].push_back(i);
    }

    for (const auto& kv : buckets) {
        int key = kv.first;
        const std::vector<size_t>& idxs = kv.second;
        int param_code = (key >> 16) & 0xF;
        bool is_4bit_m = ((key >> 15) & 1) != 0;
        bool is_quad_v = ((key >> 14) & 1) != 0;
        int bs = key & 0x3FFF;

        std::vector<TensorMetadata8Bit> host_metadata;
        host_metadata.reserve(idxs.size());
        for (size_t idx : idxs) {
            TensorMetadata8Bit meta;
            meta.param = params[idx].data_ptr();
            meta.grad = grads[idx].data_ptr();
            meta.exp_avg = exp_avgs[idx].data_ptr();
            meta.exp_avg_scales = exp_avg_scales[idx].data_ptr<float>();
            meta.exp_avg_sq = exp_avg_sqs[idx].data_ptr<uint8_t>();
            meta.exp_avg_sq_scales = exp_avg_sq_scales[idx].data_ptr<float>();
            meta.numel = params[idx].numel();
            meta.block_size = bs;
            bool apply_wd = (f_wd != 0.0f) && (!decay_norm_only || params[idx].dim() >= 2);
            meta.apply_wd = apply_wd ? 1 : 0;
            meta.pad[0] = is_quad_v ? 1 : 0;
            meta.pad[1] = meta.pad[2] = 0;
            host_metadata.push_back(meta);
        }

        const float eff_eps = (param_code == 2) ? std::max(f_eps, 6.1e-5f) : f_eps;
        const float eff_eps_scaled = (param_code == 2)
            ? static_cast<float>(std::max(eps, 6.1e-5) * bc2_sqrt)
            : eps_scaled;

        launch_multi_tensor_8bit_adamw_cuda_2d(
            host_metadata,
            param_code,
            bs,
            is_4bit_m,
            f_lr,
            f_beta1,
            f_beta2,
            eff_eps,
            f_wd,
            step_size,
            eff_eps_scaled,
            f_inv_scale,
            static_cast<uint32_t>(step),
            stochastic,
            cautious,
            stream
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step_fused", &apexw_cuda_step_fused, "ApexW CUDA Multi-Tensor Pointer-Batched Fused AdamW Step",
          py::arg("params"),
          py::arg("grads"),
          py::arg("exp_avgs"),
          py::arg("exp_avg_sqs"),
          py::arg("lr"),
          py::arg("beta1"),
          py::arg("beta2"),
          py::arg("eps"),
          py::arg("weight_decay"),
          py::arg("step") = 1,
          py::arg("decay_norm_only") = false);

    m.def("step_fused_8bit", &apexw_cuda_step_fused_8bit, "ApexW CUDA Multi-Tensor 8-bit/NF4-4bit Fused AdamW Step",
          py::arg("params"),
          py::arg("grads"),
          py::arg("exp_avgs"),
          py::arg("exp_avg_scales"),
          py::arg("exp_avg_sqs"),
          py::arg("exp_avg_sq_scales"),
          py::arg("block_sizes"),
          py::arg("lr"),
          py::arg("beta1"),
          py::arg("beta2"),
          py::arg("eps"),
          py::arg("weight_decay"),
          py::arg("step") = 1,
          py::arg("decay_norm_only") = false,
          py::arg("inv_scale") = 1.0,
          py::arg("stochastic") = true,
          py::arg("cautious") = false);

    m.def("multi_tensor_adamw_step", &apexw_cuda_step_fused, "ApexW CUDA Multi-Tensor Batched Fused AdamW Step",
          py::arg("params"),
          py::arg("grads"),
          py::arg("exp_avgs"),
          py::arg("exp_avg_sqs"),
          py::arg("lr"),
          py::arg("beta1"),
          py::arg("beta2"),
          py::arg("eps"),
          py::arg("weight_decay"),
          py::arg("step") = 1,
          py::arg("decay_norm_only") = false);

    m.def("adamw_step", &apexw_fused_adamw_cuda_step, "ApexW CUDA Single Tensor Fused AdamW Step",
          py::arg("param"),
          py::arg("grad"),
          py::arg("exp_avg"),
          py::arg("exp_avg_sq"),
          py::arg("lr"),
          py::arg("beta1"),
          py::arg("beta2"),
          py::arg("weight_decay"),
          py::arg("eps"),
          py::arg("step"));

    m.attr("_abi_version") = 2;
}
