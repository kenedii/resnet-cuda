// resnet_kernel.cu
#include <cuda_runtime.h>

// ================================================================
// General 2D Convolution (supports 1x1, 3x3, 7x7)
// ================================================================
__global__ void conv2d_kernel(
    const float *__restrict__ input,  // NCHW: [N, C, H, W]
    const float *__restrict__ weight, // [out_channels, in_channels, KH, KW]
    const float *__restrict__ bias,   // [out_channels] or nullptr
    float *__restrict__ output,       // NCHW: [N, out_C, out_H, out_W]
    int N, int C, int H, int W,
    int out_C, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w) // keep dilation=1 for ResNet
{
    // Your implementation here
}

// ================================================================
// Batch Normalization (inference + training modes)
// ================================================================
__global__ void batch_norm_kernel(
    const float *__restrict__ input,        // NCHW
    const float *__restrict__ gamma,        // [C]
    const float *__restrict__ beta,         // [C]
    const float *__restrict__ running_mean, // [C]
    const float *__restrict__ running_var,  // [C]
    float *__restrict__ output,
    float epsilon,
    int N, int C, int H, int W)
{
    // Your implementation here
}

// ================================================================
// ReLU (in-place possible)
// ================================================================
__global__ void relu_kernel(
    const float *__restrict__ input,
    float *__restrict__ output, // can be same as input for in-place
    int total_elements)
{
    // Your implementation here
}

// ================================================================
// Max Pooling (used in stem)
// ================================================================
__global__ void max_pooling_kernel(
    const float *__restrict__ input, // NCHW
    float *__restrict__ output,
    int N, int C, int H, int W,
    int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    // Your implementation here
}

// ================================================================
// Global Average Pooling / Adaptive Average Pooling (to 1x1)
// ================================================================
__global__ void global_avg_pooling_kernel(
    const float *__restrict__ input, // NCHW: [N, C, H, W]
    float *__restrict__ output,      // [N, C, 1, 1] → effectively [N, C]
    int N, int C, int H, int W)
{
    // Each thread computes ONE output value:
    // one (n, c) pair
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N*C; // Total number of output values

    if (i < total_outputs)
    {
        // Convert linear thread index into (n, c)
        int n = i / C; // which image in the batch
        int c = i % C; // which channel of that image

        float sum = 0.0f; // sum all H*W values for this (n, c)
        // Compute where this (n, c) feature map starts in memory
        // Layout: ((n * C + c) * H + h) * W + w
        int base = (n * C + c) * H * W;

        // Loop over the H*W spatial dimensions
        for (int i = 0; i < H * W; i++) {
            sum += input[base + i];
        }

        // Divide by number of elements to get the average
        output[i] = sum / (float)(H * W);
    }
}

// ================================================================
// Fully Connected (Linear) layer
// ================================================================
__global__ void fully_connected_kernel(
    const float *__restrict__ input,  // [N, in_features]
    const float *__restrict__ weight, // [out_features, in_features]
    const float *__restrict__ bias,   // [out_features] or nullptr
    float *__restrict__ output,       // [N, out_features]
    int N, int in_features, int out_features)
{
    // Your implementation here
}

// ================================================================
// Element-wise addition for residual connections
// ================================================================
__global__ void residual_add_kernel(
    const float *__restrict__ input,
    const float *__restrict__ residual, // same shape
    float *__restrict__ output,
    int total_elements)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x; // Get the sample index

    if (i < total_elements)
    {
        output[i] = input[i] + residual[i];
    }
}
