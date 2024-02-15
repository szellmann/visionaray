// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iostream>
#include <ostream>

#include <visionaray/cuda/fill.h>
#include <visionaray/math/detail/math.h>

__constant__ char cbytes[1024];

__global__ void kernel(void* ptr, size_t len, unsigned count)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= len / count)
    {
        return;
    }

    for (unsigned c = 0; c < count; ++c)
    {
        ((char*)ptr)[i * count + c] = cbytes[c];
    }
}

namespace visionaray
{
namespace cuda
{

void fill(void* ptr, size_t len, void* bytes, unsigned count)
{
    size_t num_threads = 1024;

    if (count > 1024)
    {
        std::cerr << "Fill: max. num bytes exceeded\n";
        return;
    }

    cudaMemcpyToSymbol(
            cbytes,
            bytes,
            count,
            0,
            cudaMemcpyHostToDevice
            );

    kernel<<<div_up(len / count, num_threads), num_threads>>>(ptr, len, count);
}

} // cuda
} // visionaray
