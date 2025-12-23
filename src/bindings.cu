// bindings.cu
// PyTorch CUDA extension for high-performance ResNet
// Supports both autograd and manual (pure CUDA) backpropagation modes
// Compatible with ResNet v1 (18/34) and v2 (50/101+) pre/post-activation variants

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// ===================================================================
// Forward declarations from resnet_forward.cu and resnet_backward.cu
// ===================================================================

extern void init_cuda_libs();
extern void destroy_cuda_libs();

void conv2d_cudnn(
    const float *input, const float *weight, const float *bias, float *output,
    int N, int C, int H, int W, int out_C, int KH, int KW,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w);

void batchnorm_cudnn(
    const float *input, float *output,
    const float *gamma, const float *beta,
    const float *running_mean, const float *running_var,
    float epsilon, int N, int C, int H, int W);

void relu_cudnn(float *tensor, int N, int C, int H, int W);

void conv2d_backward_cudnn(
    const float *input, const float *weight, const float *dout,
    float *dinput, float *dweight, float *dbias,
    int N, int C, int H, int W, int out_C, int KH, int KW,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w);

void batchnorm_backward_cudnn(
    const float *input, const float *dout, float *dinput,
    float *dgamma, float *dbeta,
    const float *gamma, const float *beta,
    const float *running_mean, const float *running_var,
    float epsilon, int N, int C, int H, int W);

void relu_backward_cudnn(
    const float *input, const float *dout, float *dinput,
    int N, int C, int H, int W);

// ===================================================================
// Training-mode BatchNorm (cuDNN)
// ===================================================================

void batchnorm_training_cudnn(
    const float *input, float *output,
    float *running_mean, float *running_var,
    float *save_mean, float *save_invvar,
    const float *gamma, const float *beta,
    float epsilon, float momentum,
    int N, int C, int H, int W)
{
    cudnnTensorDescriptor_t x_desc, bn_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&bn_desc);

    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnDeriveBNTensorDescriptor(bn_desc, x_desc, CUDNN_BATCHNORM_SPATIAL);

    float alpha = 1.0f, beta_bn = 0.0f;

    cudnnBatchNormalizationForwardTraining(
        at::cuda::getCurrentCUDNNHandle(),
        CUDNN_BATCHNORM_SPATIAL,
        &alpha, &beta_bn,
        x_desc, input,
        x_desc, output,
        bn_desc, gamma, beta,
        running_mean, running_var,
        momentum,
        save_mean, save_invvar,
        epsilon);

    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(bn_desc);
}

// ===================================================================
// Fused Conv + BN + ReLU Forward
// ===================================================================

torch::Tensor fused_conv_bn_relu_forward(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    const torch::Tensor &gamma,
    const torch::Tensor &beta,
    torch::Tensor &running_mean,
    torch::Tensor &running_var,
    torch::Tensor &save_mean,
    torch::Tensor &save_invvar,
    int stride, int padding,
    float eps, float momentum,
    bool training)
{
    TORCH_CHECK(input.is_contiguous() && input.scalar_type() == torch::kFloat32);

    int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int out_C = weight.size(0), KH = weight.size(2), KW = weight.size(3);
    int out_H = (H + 2 * padding - KH) / stride + 1;
    int out_W = (W + 2 * padding - KW) / stride + 1;

    auto output = torch::empty({N, out_C, out_H, out_W}, input.options());

    // Convolution
    conv2d_cudnn(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W, out_C, KH, KW,
        stride, stride, padding, padding, 1, 1);

    // BatchNorm
    if (training)
    {
        batchnorm_training_cudnn(
            output.data_ptr<float>(), output.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            save_mean.data_ptr<float>(), save_invvar.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            eps, momentum, N, out_C, out_H, out_W);
    }
    else
    {
        batchnorm_cudnn(
            output.data_ptr<float>(), output.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            eps, N, out_C, out_H, out_W);
    }

    // ReLU in-place
    relu_cudnn(output.data_ptr<float>(), N, out_C, out_H, out_W);

    return output;
}

// ===================================================================
// Fused Conv + BN Forward (no ReLU)
// ===================================================================

torch::Tensor fused_conv_bn_forward(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    const torch::Tensor &gamma,
    const torch::Tensor &beta,
    torch::Tensor &running_mean,
    torch::Tensor &running_var,
    torch::Tensor &save_mean,
    torch::Tensor &save_invvar,
    int stride, int padding,
    float eps, float momentum,
    bool training)
{
    TORCH_CHECK(input.is_contiguous() && input.scalar_type() == torch::kFloat32);

    int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int out_C = weight.size(0), KH = weight.size(2), KW = weight.size(3);
    int out_H = (H + 2 * padding - KH) / stride + 1;
    int out_W = (W + 2 * padding - KW) / stride + 1;

    auto output = torch::empty({N, out_C, out_H, out_W}, input.options());

    conv2d_cudnn(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W, out_C, KH, KW,
        stride, stride, padding, padding, 1, 1);

    if (training)
    {
        batchnorm_training_cudnn(
            output.data_ptr<float>(), output.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            save_mean.data_ptr<float>(), save_invvar.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            eps, momentum, N, out_C, out_H, out_W);
    }
    else
    {
        batchnorm_cudnn(
            output.data_ptr<float>(), output.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            eps, N, out_C, out_H, out_W);
    }

    return output;
}

// ===================================================================
// Manual Backward: Fused Conv + BN + ReLU (in-place gradient update)
// ===================================================================

void fused_conv_bn_relu_manual_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,    // original input
    const torch::Tensor &pre_relu, // output after BN, before ReLU (must be saved)
    const torch::Tensor &weight,
    const torch::Tensor &gamma,
    const torch::Tensor &save_mean,
    const torch::Tensor &save_invvar,
    torch::Tensor &grad_input,
    torch::Tensor &grad_weight,
    torch::Tensor &grad_bias,
    torch::Tensor &grad_gamma,
    torch::Tensor &grad_beta,
    int stride, int padding)
{
    TORCH_CHECK(grad_output.is_contiguous());

    int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int out_C = weight.size(0), KH = weight.size(2), KW = weight.size(3);
    int out_H = grad_output.size(2), out_W = grad_output.size(3);

    // Temporary gradient buffer after ReLU backward
    auto grad_bn = grad_output.contiguous();

    // 1. ReLU backward
    relu_backward_cudnn(pre_relu.data_ptr<float>(), grad_bn.data_ptr<float>(),
                        grad_bn.data_ptr<float>(), N, out_C, out_H, out_W);

    // 2. BatchNorm backward
    batchnorm_backward_cudnn(
        pre_relu.data_ptr<float>(), // input to BN (post-conv, pre-relu)
        grad_bn.data_ptr<float>(),
        grad_bn.data_ptr<float>(), // dinput (overwritten)
        grad_gamma.data_ptr<float>(),
        grad_beta.data_ptr<float>(),
        gamma.data_ptr<float>(),
        nullptr, // beta not needed
        save_mean.data_ptr<float>(),
        save_invvar.data_ptr<float>(),
        1e-5, N, out_C, out_H, out_W);

    // 3. Conv backward
    conv2d_backward_cudnn(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        grad_bn.data_ptr<float>(),
        grad_input.defined() ? grad_input.data_ptr<float>() : nullptr,
        grad_weight.defined() ? grad_weight.data_ptr<float>() : nullptr,
        grad_bias.defined() ? grad_bias.data_ptr<float>() : nullptr,
        N, C, H, W, out_C, KH, KW,
        stride, stride, padding, padding, 1, 1);
}

// ===================================================================
// Manual Backward: Fused Conv + BN (no ReLU)
// ===================================================================

void fused_conv_bn_manual_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &post_bn, // output after BN
    const torch::Tensor &weight,
    const torch::Tensor &gamma,
    const torch::Tensor &save_mean,
    const torch::Tensor &save_invvar,
    torch::Tensor &grad_input,
    torch::Tensor &grad_weight,
    torch::Tensor &grad_bias,
    torch::Tensor &grad_gamma,
    torch::Tensor &grad_beta,
    int stride, int padding)
{
    int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int out_C = weight.size(0), KH = weight.size(2), KW = weight.size(3);
    int out_H = grad_output.size(2), out_W = grad_output.size(3);

    auto grad_bn = grad_output.contiguous();

    batchnorm_backward_cudnn(
        post_bn.data_ptr<float>(),
        grad_bn.data_ptr<float>(),
        grad_bn.data_ptr<float>(),
        grad_gamma.data_ptr<float>(),
        grad_beta.data_ptr<float>(),
        gamma.data_ptr<float>(),
        nullptr,
        save_mean.data_ptr<float>(),
        save_invvar.data_ptr<float>(),
        1e-5, N, out_C, out_H, out_W);

    conv2d_backward_cudnn(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        grad_bn.data_ptr<float>(),
        grad_input.defined() ? grad_input.data_ptr<float>() : nullptr,
        grad_weight.defined() ? grad_weight.data_ptr<float>() : nullptr,
        grad_bias.defined() ? grad_bias.data_ptr<float>() : nullptr,
        N, C, H, W, out_C, KH, KW,
        stride, stride, padding, padding, 1, 1);
}

// ===================================================================
// PYBIND11 Bindings
// ===================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_conv_bn_relu_forward", &fused_conv_bn_relu_forward,
          "Fused Conv+BN+ReLU forward (supports training/inference)");

    m.def("fused_conv_bn_forward", &fused_conv_bn_forward,
          "Fused Conv+BN forward (no ReLU, for bottleneck projection)");

    m.def("fused_conv_bn_relu_manual_backward", &fused_conv_bn_relu_manual_backward,
          "Manual backward for fused Conv+BN+ReLU (pure CUDA, no autograd)");

    m.def("fused_conv_bn_manual_backward", &fused_conv_bn_manual_backward,
          "Manual backward for fused Conv+BN (no ReLU)");

    m.def("init_libs", []()
          { init_cuda_libs(); });
    m.def("destroy_libs", []()
          { destroy_cuda_libs(); });
}
