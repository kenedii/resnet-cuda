// ================================================================
// resnet_forward.cu
// cuBLAS + cuDNN v8 compatible kernels for ResNet
// ================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <assert.h>
#include <stdio.h>

// ================================================================
// Global cuBLAS / cuDNN handles
// ================================================================

static cublasHandle_t g_cublas = nullptr;
static cudnnHandle_t g_cudnn = nullptr;

// ================================================================
// Library lifecycle
// ================================================================

void init_cuda_libs()
{
    cublasCreate(&g_cublas);
    cudnnCreate(&g_cudnn);
}

void destroy_cuda_libs()
{
    if (g_cublas)
        cublasDestroy(g_cublas);
    if (g_cudnn)
        cudnnDestroy(g_cudnn);
}

// ================================================================
// Conv2D (cuDNN v8-safe, heuristic-based)
// ================================================================

void conv2d_cudnn(
    const float *input,  // NCHW
    const float *weight, // [out_C, C, KH, KW]
    const float *bias,   // [out_C] or nullptr
    float *output,       // NCHW
    int N, int C, int H, int W,
    int out_C, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w)
{
    cudnnTensorDescriptor_t in_desc, out_desc, bias_desc;
    cudnnFilterDescriptor_t filt_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, C, H, W);

    cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out_C, C, KH, KW);

    cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT);

    int outN, outC, outH, outW;
    cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &outN, &outC, &outH, &outW);

    cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        outN, outC, outH, outW);

    cudnnSetTensor4dDescriptor(
        bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        1, outC, 1, 1);

    // -------------------------------
    // v8-safe algorithm selection
    // -------------------------------
    cudnnConvolutionFwdAlgoPerf_t perf;
    int returned = 0;

    cudnnGetConvolutionForwardAlgorithm_v7(
        g_cudnn,
        in_desc,
        filt_desc,
        conv_desc,
        out_desc,
        1,
        &returned,
        &perf);

    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        g_cudnn,
        in_desc,
        filt_desc,
        conv_desc,
        out_desc,
        perf.algo,
        &workspace_bytes);

    void *workspace = nullptr;
    if (workspace_bytes > 0)
        cudaMalloc(&workspace, workspace_bytes);

    float alpha = 1.0f;
    float beta0 = 0.0f;

    cudnnConvolutionForward(
        g_cudnn,
        &alpha,
        in_desc, input,
        filt_desc, weight,
        conv_desc,
        perf.algo,
        workspace, workspace_bytes,
        &beta0,
        out_desc, output);

    if (bias)
    {
        cudnnAddTensor(
            g_cudnn,
            &alpha,
            bias_desc, bias,
            &alpha,
            out_desc, output);
    }

    if (workspace)
        cudaFree(workspace);

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
}

// ================================================================
// Batch Normalization (Inference)
// ================================================================

void batchnorm_cudnn(
    const float *input,
    float *output,
    const float *gamma,
    const float *beta,
    const float *running_mean,
    const float *running_var,
    float epsilon,
    int N, int C, int H, int W)
{
    cudnnTensorDescriptor_t x_desc, bn_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&bn_desc);

    cudnnSetTensor4dDescriptor(
        x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, C, H, W);

    cudnnDeriveBNTensorDescriptor(
        bn_desc,
        x_desc,
        CUDNN_BATCHNORM_SPATIAL);

    float alpha = 1.0f;
    float beta0 = 0.0f;

    cudnnBatchNormalizationForwardInference(
        g_cudnn,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha,
        &beta0,
        x_desc, input,
        x_desc, output,
        bn_desc,
        gamma,
        beta,
        running_mean,
        running_var,
        epsilon);

    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(bn_desc);
}

// ================================================================
// ReLU (cuDNN)
// ================================================================

void relu_cudnn(float *tensor, int N, int C, int H, int W)
{
    cudnnTensorDescriptor_t desc;
    cudnnActivationDescriptor_t act;

    cudnnCreateTensorDescriptor(&desc);
    cudnnCreateActivationDescriptor(&act);

    cudnnSetTensor4dDescriptor(
        desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, C, H, W);

    cudnnSetActivationDescriptor(
        act,
        CUDNN_ACTIVATION_RELU,
        CUDNN_PROPAGATE_NAN,
        0.0);

    float alpha = 1.0f;
    float beta0 = 0.0f;

    cudnnActivationForward(
        g_cudnn,
        act,
        &alpha,
        desc, tensor,
        &beta0,
        desc, tensor);

    cudnnDestroyActivationDescriptor(act);
    cudnnDestroyTensorDescriptor(desc);
}

// ================================================================
// Max Pooling (cuDNN)
// ================================================================

void maxpool_cudnn(
    const float *input,
    float *output,
    int N, int C, int H, int W,
    int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnPoolingDescriptor_t pool_desc;

    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreatePoolingDescriptor(&pool_desc);

    cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, C, H, W);

    cudnnSetPooling2dDescriptor(
        pool_desc,
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,
        KH, KW,
        pad_h, pad_w,
        stride_h, stride_w);

    int outN, outC, outH, outW;
    cudnnGetPooling2dForwardOutputDim(
        pool_desc, in_desc,
        &outN, &outC, &outH, &outW);

    cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        outN, outC, outH, outW);

    float alpha = 1.0f;
    float beta0 = 0.0f;

    cudnnPoolingForward(
        g_cudnn,
        pool_desc,
        &alpha,
        in_desc, input,
        &beta0,
        out_desc, output);

    cudnnDestroyPoolingDescriptor(pool_desc);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
}

// ================================================================
// Fully Connected (cuBLAS GEMM)
// ================================================================

void fully_connected_cublas(
    const float *input,  // [N, in_features]
    const float *weight, // [out_features, in_features]
    const float *bias,   // [out_features] or nullptr
    float *output,       // [N, out_features]
    int N, int in_features, int out_features)
{
    float alpha = 1.0f;
    float beta0 = 0.0f;

    // output = input * weight^T
    cublasSgemm(
        g_cublas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_features,
        N,
        in_features,
        &alpha,
        weight, in_features,
        input, in_features,
        &beta0,
        output, out_features);

    // Bias add intentionally left separate
}

// ================================================================
// Residual Add
// ================================================================

__global__ void residual_add_kernel(
    const float *input,
    const float *residual,
    float *output,
    int total_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements)
    {
        output[i] = input[i] + residual[i];
    }
}
