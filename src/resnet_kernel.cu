// resnet_kernel.cu
#include <cuda_runtime.h>
#include <float.h>

// ================================================================
// General 2D Convolution (supports 1x1, 3x3, 7x7)
// ================================================================
__global__ void conv2d_kernel(
    const float *__restrict__ input,  // NCHW
    const float *__restrict__ weight, // [out_C, C, KH, KW]
    const float *__restrict__ bias,   // [out_C] or nullptr
    float *__restrict__ output,       // NCHW
    int N, int C, int H, int W,
    int out_C, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w) // usually = 1
{
    // Compute linear thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute output spatial size
    int out_H = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) / stride_h + 1;
    int out_W = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) / stride_w + 1;

    // Total number of output elements
    int total_outputs = N * out_C * out_H * out_W;

    // Check bounds
    if (idx < total_outputs)
    {
        // ------------------------------------------------------------
        // Decode linear index -> (n, out_c, out_h, out_w)
        // ------------------------------------------------------------
        int out_w = idx % out_W;
        int out_h = (idx / out_W) % out_H;
        int out_c = (idx / (out_H * out_W)) % out_C;
        int n = idx / (out_C * out_H * out_W);

        // Initialize accumulator
        float sum = 0.0f;

        // ------------------------------------------------------------
        // Convolution loop
        // ------------------------------------------------------------
        for (int c = 0; c < C; ++c)
        {
            for (int kh = 0; kh < KH; ++kh)
            {
                for (int kw = 0; kw < KW; ++kw)
                {
                    // Compute input coordinates
                    int in_h = out_h * stride_h - pad_h + kh * dilation_h;
                    int in_w = out_w * stride_w - pad_w + kw * dilation_w;
                    // Check input bounds (padding handling)
                    if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W)
                    {
                        // Input index: ((n * C + c) * H + in_h) * W + in_w
                        int input_idx =
                            ((n * C + c) * H + in_h) * W + in_w;

                        // Weight index:
                        // (((out_c * C + c) * KH + kh) * KW + kw)
                        int weight_idx =
                            ((out_c * C + c) * KH + kh) * KW + kw;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Add bias if present
        if (bias != nullptr)
        {
            sum += bias[out_c];
        }

        // Write output
        output[idx] = sum;
    }
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
    // Compute global linear index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of elements in the tensor
    int total_elements = N * C * H * W;

    // Make sure thread index is valid
    if (idx < total_elements)
    {
        // Recover channel index from flattened NCHW index
        // idx = ((n * C + c) * H + h) * W + w
        int hw = H * W;
        int c = (idx / hw) % C;

        // Load input value
        float x = input[idx];

        // Load batchnorm parameters for this channel
        float mean = running_mean[c];
        float var = running_var[c];
        float g = gamma[c];
        float b = beta[c];

        // Normalize
        float x_hat = (x - mean) * rsqrtf(var + epsilon);

        // Scale and shift
        output[idx] = g * x_hat + b;
    }
}

// ================================================================
// ReLU (in-place possible)
// ================================================================
__global__ void relu_kernel(
    const float *__restrict__ input,
    float *__restrict__ output, // can be same as input for in-place
    int total_elements)
{
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't go out of bounds
    if (idx < total_elements)
    {
        float x = input[idx];

        // ReLU operation: max(0, x)
        output[idx] = (x > 0.0f) ? x : 0.0f;
    }
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

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * C * ((H + 2 * pad_h - KH) / stride_h + 1) * ((W + 2 * pad_w - KW) / stride_w + 1);

    if (i < total_outputs)
    {
        // Convert linear thread index into (n, c, out_h, out_w)
        int out_w = i % ((W + 2 * pad_w - KW) / stride_w + 1);
        int out_h = (i / ((W + 2 * pad_w - KW) / stride_w + 1)) % ((H + 2 * pad_h - KH) / stride_h + 1);
        int c = (i / (((W + 2 * pad_w - KW) / stride_w + 1) * ((H + 2 * pad_h - KH) / stride_h + 1))) % C;
        int n = i / (C * ((H + 2 * pad_h - KH) / stride_h + 1) * ((W + 2 * pad_w - KW) / stride_w + 1));

        float max_val = -FLT_MAX;

        // Compute the start and end indices of the pooling window
        int h_start = out_h * stride_h - pad_h;
        int w_start = out_w * stride_w - pad_w;
        int h_end = min(h_start + KH, H);
        int w_end = min(w_start + KW, W);
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);

        // Loop over the pooling window
        for (int h = h_start; h < h_end; ++h)
        {
            for (int w = w_start; w < w_end; ++w)
            {
                int input_idx = ((n * C + c) * H + h) * W + w;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }

        // Write the result to output
        int output_idx = ((n * C + c) * ((H + 2 * pad_h - KH) / stride_h + 1) + out_h) * ((W + 2 * pad_w - KW) / stride_w + 1) + out_w;
        output[output_idx] = max_val;
    }
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
    int total_outputs = N * C; // Total number of output values

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
        for (int i = 0; i < H * W; i++)
        {
            sum += input[base + i];
        }

        // Divide by number of elements to get the average
        output[i] = sum / (float)(H * W);
    }
}

// ================================================================
// Fully Connected (Linear) layer
__global__ void fully_connected_kernel(
    const float *__restrict__ input,  // [N, in_features]
    const float *__restrict__ weight, // [out_features, in_features]
    const float *__restrict__ bias,   // [out_features] or nullptr
    float *__restrict__ output,       // [N, out_features]
    int N, int in_features, int out_features)
{
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of output elements
    int total_outputs = N * out_features;

    // Make sure this thread maps to a valid output
    if (idx < total_outputs)
    {
        // Convert linear index -> (n, o)
        int n = idx / out_features; // which input sample
        int o = idx % out_features; // which output neuron

        float sum = 0.0f;

        // Compute memory offsets
        int input_base = n * in_features;
        int weight_base = o * in_features;

        // Dot product over input features
        for (int i = 0; i < in_features; ++i)
        {
            sum += input[input_base + i] * weight[weight_base + i];
        }

        // Add bias if provided
        if (bias != nullptr)
        {
            sum += bias[o];
        }

        // Write output
        output[idx] = sum;
    }
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
