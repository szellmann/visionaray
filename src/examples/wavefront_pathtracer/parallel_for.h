// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_EX_WAVEFRONT_PATHTRACER_PARALLEL_FOR_H
#define VSNRAY_EX_WAVEFRONT_PATHTRACER_PARALLEL_FOR_H 1

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace visionaray
{

#ifdef __CUDACC__

namespace detail
{

template <typename Func>
__global__ void kernel(Func func, int n)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        func(i);
    }
}

template <typename Sampler, typename Func>
__global__ void kernel(Sampler /* */, Func func, int n)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        Sampler samp(cuda_seed());

        func(i, samp);
    }
}

} // detail

template <typename Func>
void parallel_for(int first, int last, Func func)
{
    int n = last - first;
    int block_size = 64;
    int grid_size = div_up(n, block_size);
    detail::kernel<<<grid_size, block_size>>>(func, n);
}

template <typename Sampler, typename Func>
void parallel_for(Sampler /* */, int first, int last, Func func)
{
    int n = last - first;
    int block_size = 64;
    int grid_size = div_up(n, block_size);
    detail::kernel<<<grid_size, block_size>>>(Sampler{}, func, n);
}

#else

template <typename Func>
void parallel_for(int first, int last, Func func)
{
    tbb::parallel_for(
        tbb::blocked_range<int>(first, last),
        [&](tbb::blocked_range<int> r)
        {
            for (auto it = r.begin(); it != r.end(); ++it)
            {
                func(it);
            }
        }
        );
}

template <typename Sampler, typename Func>
void parallel_for(Sampler /* */, int first, int last, Func func)
{
    tbb::parallel_for(
        tbb::blocked_range<int>(first, last),
        [&](tbb::blocked_range<int> r)
        {
            Sampler samp(detail::tic());
            for (auto it = r.begin(); it != r.end(); ++it)
            {
                func(it, samp);
            }
        }
        );
}

#endif

} // visionaray

#endif // VSNRAY_EX_WAVEFRONT_PATHTRACER_PARALLEL_FOR_H
