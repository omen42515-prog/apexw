#include <torch/extension.h>
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

    const float f_lr = static_cast<float>(lr);
    const float f_beta1 = static_cast<float>(beta1);
    const float f_beta2 = static_cast<float>(beta2);
    const float f_eps = static_cast<float>(eps);
    const float f_wd = static_cast<float>(weight_decay);

    // Bias corrections precomputed on CPU once per step
    const float bias_correction1 = 1.0f - std::pow(f_beta1, static_cast<float>(step));
    const float bias_correction2 = 1.0f - std::pow(f_beta2, static_cast<float>(step));
    const float step_size = f_lr / bias_correction1;
    const float bias_correction2_sqrt = std::sqrt(bias_correction2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(params[0].get_device()).stream();

    // Group tensors by (param_dtype, state_dtype) to support mixed precision models
    // In standard models, 99.9% of tensors share the same dtypes and execute in 1 group
    std::unordered_map<int, std::vector<size_t>> dtype_groups;
    for (size_t i = 0; i < num_tensors; ++i) {
        torch::Tensor& p = params[i];
        const torch::Tensor& g = grads[i];
        torch::Tensor& m = exp_avgs[i];
        torch::Tensor& v = exp_avg_sqs[i];

        TORCH_CHECK(p.is_cuda() && g.is_cuda() && m.is_cuda() && v.is_cuda(),
                    "All tensors must be CUDA tensors");
        TORCH_CHECK(p.is_contiguous() && g.is_contiguous() && m.is_contiguous() && v.is_contiguous(),
                    "All tensors must be contiguous");

        int param_code = get_dtype_code(p.scalar_type());
        int state_code = get_dtype_code(m.scalar_type());

        TORCH_CHECK(param_code != -1, "ApexW CUDA supports Float32, BFloat16, and Float16 parameters");
        TORCH_CHECK(state_code != -1, "ApexW CUDA supports Float32, BFloat16, and Float16 optimizer states");

        int key = (param_code << 4) | state_code;
        dtype_groups[key].push_back(i);
    }

    for (const auto& kv : dtype_groups) {
        int key = kv.first;
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
        }

        if (max_numel > 0) {
            launch_multi_tensor_adamw_cuda_2d(
                host_metadata,
                max_numel,
                param_code,
                state_code,
                f_lr,
                f_beta1,
                f_beta2,
                f_eps,
                step_size,
                bias_correction2_sqrt,
                static_cast<uint32_t>(step),
                stream
            );
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
    TORCH_CHECK(param.is_contiguous() && grad.is_contiguous() && exp_avg.is_contiguous() && exp_avg_sq.is_contiguous(),
                "All tensors must be contiguous");

    int param_code = get_dtype_code(param.scalar_type());
    int state_code = get_dtype_code(exp_avg.scalar_type());

    TORCH_CHECK(param_code != -1, "ApexW CUDA supports Float32, BFloat16, and Float16 parameters");
    TORCH_CHECK(state_code != -1, "ApexW CUDA supports Float32, BFloat16, and Float16 optimizer states");
    TORCH_CHECK(exp_avg.scalar_type() == exp_avg_sq.scalar_type(), "exp_avg and exp_avg_sq must share same dtype");

    const int64_t total_elements = param.numel();
    if (total_elements == 0) return;

    const float f_lr = static_cast<float>(lr);
    const float f_beta1 = static_cast<float>(beta1);
    const float f_beta2 = static_cast<float>(beta2);
    const float f_wd = static_cast<float>(weight_decay);
    const float f_eps = static_cast<float>(eps);

    const float bias_correction1 = 1.0f - std::pow(f_beta1, static_cast<float>(step));
    const float bias_correction2 = 1.0f - std::pow(f_beta2, static_cast<float>(step));
    const float step_size = f_lr / bias_correction1;
    const float bias_correction2_sqrt = std::sqrt(bias_correction2);

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
        f_eps,
        step_size,
        bias_correction2_sqrt,
        static_cast<uint32_t>(step),
        stream
    );
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
}
