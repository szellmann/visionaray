// This file is distributed under the MIT license.
// See the LICENSE file for details.
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.

// HIP test exercising visionaray::random_generator on the device.
//
// Constructs random_generator<float> inside a kernel and pulls samples,
// verifying the thrust-backed device RNG path is actually compiled and
// produces finite, varied values in [0, 1).

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>

#include <visionaray/random_generator.h>
#include <visionaray/hip/device_vector.h>
#include <visionaray/hip/safe_call.h>

using namespace visionaray;

__global__ void rng_kernel(float* out, int samples_per_thread, int n_threads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads)
    {
        return;
    }

    // Per-thread distinct seed; constructed and consumed entirely on device.
    random_generator<float> gen(static_cast<unsigned>(tid) * 9781u + 1u);

    for (int s = 0; s < samples_per_thread; ++s)
    {
        out[tid * samples_per_thread + s] = gen.next();
    }
}

int main()
{
    std::cout << "Testing visionaray device random_generator (HIP)..." << std::endl;

    hipDeviceProp_t props;
    HIP_SAFE_CALL(hipGetDeviceProperties(&props, 0));
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "Warp size: " << props.warpSize << std::endl;

    const int n_threads = 4096;
    const int samples_per_thread = 8;
    const int total = n_threads * samples_per_thread;

    hip::device_vector<float> d_out(total);

    int block_size = 256;
    int grid_size = (n_threads + block_size - 1) / block_size;
    rng_kernel<<<grid_size, block_size>>>(d_out.data(), samples_per_thread, n_threads);
    HIP_SAFE_CALL(hipGetLastError());
    HIP_SAFE_CALL(hipDeviceSynchronize());

    std::vector<float> h_out(total);
    HIP_SAFE_CALL(hipMemcpy(h_out.data(), d_out.data(), total * sizeof(float), hipMemcpyDeviceToHost));

    bool all_finite = true;
    bool all_in_range = true;
    double sum = 0.0;
    float vmin = h_out[0];
    float vmax = h_out[0];
    for (int i = 0; i < total; ++i)
    {
        float v = h_out[i];
        if (!std::isfinite(v))
        {
            all_finite = false;
        }
        if (v < 0.0f || v >= 1.0f)
        {
            all_in_range = false;
        }
        sum += v;
        vmin = v < vmin ? v : vmin;
        vmax = v > vmax ? v : vmax;
    }
    double mean = sum / total;

    // "Varied" guard: a broken/degenerate RNG would emit a constant, so the
    // spread must be wide and the mean near 0.5 for a uniform[0,1) draw.
    bool varied = (vmax - vmin) > 0.5f;
    bool mean_ok = mean > 0.3 && mean < 0.7;

    std::cout << "samples=" << total
              << " min=" << vmin
              << " max=" << vmax
              << " mean=" << mean << std::endl;

    if (all_finite && all_in_range && varied && mean_ok)
    {
        std::cout << "PASS: device random_generator produced finite, varied values" << std::endl;
        return 0;
    }

    std::cerr << "FAIL: finite=" << all_finite
              << " in_range=" << all_in_range
              << " varied=" << varied
              << " mean_ok=" << mean_ok << std::endl;
    return 1;
}
