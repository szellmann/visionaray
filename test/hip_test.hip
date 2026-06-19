// This file is distributed under the MIT license.
// See the LICENSE file for details.
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.

// HIP test to verify visionaray HIP headers compile and run

#include <iostream>
#include <cstdlib>

#include <hip/hip_runtime.h>
#include <visionaray/scheduler.h>
#include <visionaray/math/vector.h>
#include <visionaray/hip/device_vector.h>
#include <visionaray/hip/safe_call.h>

using namespace visionaray;

// Simple kernel to verify GPU code runs
__global__ void add_kernel(float* a, float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    std::cout << "Testing visionaray HIP support..." << std::endl;

    // Check device
    hipDeviceProp_t props;
    HIP_SAFE_CALL(hipGetDeviceProperties(&props, 0));
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "Warp size: " << props.warpSize << std::endl;

    // Test hip::device_vector
    const int N = 1024;
    hip::device_vector<float> d_a(N);
    hip::device_vector<float> d_b(N);
    hip::device_vector<float> d_c(N);

    // Allocate host data
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Copy to device
    HIP_SAFE_CALL(hipMemcpy(d_a.data(), h_a.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_SAFE_CALL(hipMemcpy(d_b.data(), h_b.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Launch kernel
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size>>>(d_a.data(), d_b.data(), d_c.data(), N);

    HIP_SAFE_CALL(hipDeviceSynchronize());

    // Copy result back
    HIP_SAFE_CALL(hipMemcpy(h_c.data(), d_c.data(), N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify result
    bool pass = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            std::cerr << "Mismatch at " << i << ": expected " << expected << ", got " << h_c[i] << std::endl;
            pass = false;
            break;
        }
    }

    if (pass) {
        std::cout << "PASS: Basic HIP test succeeded" << std::endl;
        return 0;
    } else {
        std::cerr << "FAIL: Basic HIP test failed" << std::endl;
        return 1;
    }
}
