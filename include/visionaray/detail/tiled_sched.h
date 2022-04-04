// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TILED_SCHED_H
#define VSNRAY_DETAIL_TILED_SCHED_H 1

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

static thread_local int2 threadIdx;
static thread_local int2 blockIdx;
static thread_local int2 blockDim;
static thread_local int2 gridDim;

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
        blockDim.x = tr.rows().tile_size();
        blockDim.y = tr.cols().tile_size();

        gridDim.x = tr.rows().length();
        gridDim.y = tr.cols().length();

        visionaray::parallel_for(
            pool_,
            tr,
            [=](range2d<int> const& r)
            {
                blockIdx.x = r.rows().begin() / r.rows().length();
                blockIdx.y = r.cols().begin() / r.cols().length();

                for (int y = r.cols().begin(); y < r.cols().end(); y += packet_height)
                {
                    for (int x = r.rows().begin(); x < r.rows().end(); x += packet_width)
                    {
                        threadIdx.x = x % r.rows().length();
                        threadIdx.y = y % r.cols().length();

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
