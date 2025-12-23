// ================================================================
// resnet_backward.cu
// cuBLAS + cuDNN optimized backward kernels for ResNet
// ================================================================
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <assert.h>
#include <stdio.h>

// ================================================================
// Global cuBLAS / cuDNN handles (defined in resnet_forward.cu)
// ================================================================
extern cublasHandle_t g_cublas;
extern cudnnHandle_t g_cudnn;

// ================================================================
// Bias gradient helper kernel — MOVED UP to fix "undefined" error
// ================================================================
__global__ void add_bias_grad(const float *dout, float *dbias, int N, int out_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_features)
    {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n)
        {
            sum += dout[n * out_features + idx];
        }
        dbias[idx] = sum;
    }
}

// ================================================================
// Conv2D Backward (cuDNN) - Compatible with cuDNN 8.x
// ================================================================
void conv2d_backward_cudnn(
    const float *input,
    const float *weight,
    const float *dout,
    float *dinput,
    float *dweight,
    float *dbias,
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

    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_C, C, KH, KW);
    cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w,
                                    dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    int out_N, out_C_dim, out_H, out_W;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc,
                                          &out_N, &out_C_dim, &out_H, &out_W);

    assert(out_N == N && out_C_dim == out_C);

    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               N, out_C, out_H, out_W);

    if (dbias)
    {
        cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, out_C, 1, 1);
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    void *workspace = nullptr;
    size_t workspace_bytes = 0;

    if (dinput)
    {
        cudnnConvolutionBwdDataAlgo_t bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

        cudnnGetConvolutionBackwardDataWorkspaceSize(g_cudnn, filt_desc, out_desc, conv_desc,
                                                     in_desc, bwd_data_algo, &workspace_bytes);

        if (workspace_bytes > 0)
            cudaMalloc(&workspace, workspace_bytes);

        cudnnConvolutionBackwardData(g_cudnn, &alpha,
                                     filt_desc, weight,
                                     out_desc, dout,
                                     conv_desc, bwd_data_algo,
                                     workspace, workspace_bytes,
                                     &beta, in_desc, dinput);

        if (workspace_bytes > 0)
        {
            cudaFree(workspace);
            workspace = nullptr;
        }
    }

    if (dweight)
    {
        cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

        cudnnGetConvolutionBackwardFilterWorkspaceSize(g_cudnn, in_desc, out_desc, conv_desc,
                                                       filt_desc, bwd_filter_algo, &workspace_bytes);

        if (workspace_bytes > 0)
            cudaMalloc(&workspace, workspace_bytes);

        cudnnConvolutionBackwardFilter(g_cudnn, &alpha,
                                       in_desc, input,
                                       out_desc, dout,
                                       conv_desc, bwd_filter_algo,
                                       workspace, workspace_bytes,
                                       &beta, filt_desc, dweight);

        if (workspace_bytes > 0)
            cudaFree(workspace);
    }

    if (dbias)
    {
        cudnnConvolutionBackwardBias(g_cudnn, &alpha, out_desc, dout,
                                     &beta, bias_desc, dbias);
    }

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    if (dbias)
        cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
}

// ================================================================
// Batch Normalization Backward
// ================================================================
void batchnorm_backward_cudnn(
    const float *input,
    const float *dout,
    float *dinput,
    float *dgamma,
    float *dbeta,
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

    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetTensor4dDescriptor(bn_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1);

    float alpha_data = 1.0f, beta_data = 0.0f;
    float alpha_param = 1.0f, beta_param = 0.0f;

    cudnnBatchNormalizationBackward(
        g_cudnn,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha_data, &beta_data,
        &alpha_param, &beta_param,
        x_desc, input,
        x_desc, dout,
        x_desc, dinput,
        bn_desc, gamma, dgamma, dbeta,
        epsilon,
        nullptr, nullptr);

    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(bn_desc);
}

// ================================================================
// ReLU Backward (cuDNN)
// ================================================================
void relu_backward_cudnn(
    const float *input,
    const float *dout,
    float *dinput,
    int N, int C, int H, int W)
{
    cudnnTensorDescriptor_t desc;
    cudnnActivationDescriptor_t act;

    cudnnCreateTensorDescriptor(&desc);
    cudnnCreateActivationDescriptor(&act);

    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

    float alpha = 1.0f;
    float beta = 0.0f;

    cudnnActivationBackward(g_cudnn, act,
                            &alpha, desc, input,
                            desc, dout,
                            desc, input,
                            &beta, desc, dinput);

    cudnnDestroyActivationDescriptor(act);
    cudnnDestroyTensorDescriptor(desc);
}

// ================================================================
// Max Pooling Backward (cuDNN)
// ================================================================
void maxpool_backward_cudnn(
    const float *input,
    const float *dout,
    float *dinput,
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

    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                KH, KW, pad_h, pad_w, stride_h, stride_w);

    int out_N, out_C, out_H, out_W;
    cudnnGetPooling2dForwardOutputDim(pool_desc, in_desc, &out_N, &out_C, &out_H, &out_W);

    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               N, C, out_H, out_W);

    float alpha = 1.0f;
    float beta = 0.0f;

    cudnnPoolingBackward(g_cudnn, pool_desc, &alpha,
                         out_desc, nullptr,
                         out_desc, dout,
                         in_desc, input,
                         &beta,
                         in_desc, dinput);

    cudnnDestroyPoolingDescriptor(pool_desc);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
}

// ================================================================
// Fully Connected Backward (cuBLAS GEMM)
// ================================================================
void fully_connected_backward_cublas(
    const float *input,
    const float *weight,
    const float *dout,
    float *dinput,
    float *dweight,
    float *dbias,
    int N, int in_features, int out_features)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    if (dinput)
    {
        cublasSgemm(g_cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    in_features, N, out_features,
                    &alpha,
                    weight, in_features,
                    dout, out_features,
                    &beta,
                    dinput, in_features);
    }

    if (dweight)
    {
        cublasSgemm(g_cublas,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    out_features, in_features, N,
                    &alpha,
                    dout, out_features,
                    input, in_features,
                    &beta,
                    dweight, out_features);
    }

    if (dbias)
    {
        int threads = 256;
        int blocks = (out_features + threads - 1) / threads;
        add_bias_grad<<<blocks, threads>>>(dout, dbias, N, out_features);
    }
}

// ================================================================
// Residual Add Backward
// ================================================================
__global__ void residual_add_backward_kernel(
    const float *dout,
    float *dinput1,
    float *dinput2,
    int total_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements)
    {
        float grad = dout[i];
        if (dinput1)
            dinput1[i] += grad;
        if (dinput2)
            dinput2[i] += grad;
    }
}
