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
        visionaray::parallel_for(
            pool_,
            tr,
            [=](range2d<int> const& r)
            {
                for (int y = r.cols().begin(); y < r.cols().end(); y += packet_height)
                {
                    for (int x = r.rows().begin(); x < r.rows().end(); x += packet_width)
                    {
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
