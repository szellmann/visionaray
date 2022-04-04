// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TILED_SCHED_H
#define VSNRAY_DETAIL_TILED_SCHED_H 1

#include "../math/detail/math.h"
#include "basic_sched.h"
#include "parallel_for.h"
#include "range.h"
#include "thread_pool.h"

namespace visionaray
{

struct int2
{
    int x;
    int y;
};

// These are enabled by the user, who is supposed to
// define the flag
// VSNRAY_TILED_SCHED_CUDA_STYLE_THREAD_INTROSPECTION
// before including headers, if they desire to use this
// feature
#ifdef VSNRAY_TILED_SCHED_CUDA_STYLE_THREAD_INTROSPECTION
static thread_local int2 threadIdx;
static thread_local int2 blockIdx;
static thread_local int2 launchIdx;
static int2 blockDim;
static int2 gridDim;
static int2 launchDim;
#endif

struct tiled_sched_backend
{
    explicit tiled_sched_backend(unsigned num_threads)
        : pool_(num_threads)
    {
    }

    void reset(unsigned num_threads)
    {
        pool_.reset(num_threads);
    }

    template <typename Func>
    void for_each_packet(
            tiled_range2d<int> const& tr,
            int packet_width,
            int packet_height,
            Func const& func
            )
    {
#ifdef VSNRAY_TILED_SCHED_CUDA_STYLE_THREAD_INTROSPECTION
        blockDim.x = tr.rows().tile_size();
        blockDim.y = tr.cols().tile_size();

        gridDim.x = div_up(tr.rows().length(), tr.rows().tile_size());
        gridDim.y = div_up(tr.cols().length(), tr.cols().tile_size());

        launchDim.x = tr.rows().length();
        launchDim.y = tr.cols().length();
#endif

        visionaray::parallel_for(
            pool_,
            tr,
            [=](range2d<int> const& r)
            {
#ifdef VSNRAY_TILED_SCHED_CUDA_STYLE_THREAD_INTROSPECTION
                blockIdx.x = r.rows().begin() / r.rows().length();
                blockIdx.y = r.cols().begin() / r.cols().length();
#endif

                for (int y = r.cols().begin(); y < r.cols().end(); y += packet_height)
                {
                    for (int x = r.rows().begin(); x < r.rows().end(); x += packet_width)
                    {
#ifdef VSNRAY_TILED_SCHED_CUDA_STYLE_THREAD_INTROSPECTION
                        threadIdx.x = x % r.rows().length();
                        threadIdx.y = y % r.cols().length();

                        launchIdx.x = x;
                        launchIdx.y = y;
#endif

                        func(x, y);
                    }
                }
            });
    }

    thread_pool pool_;
};

template <typename R>
using tiled_sched = basic_sched<tiled_sched_backend, R>;

} // visionaray

#endif // VSNRAY_DETAIL_TILED_SCHED
