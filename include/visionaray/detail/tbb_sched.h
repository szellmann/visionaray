// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TBB_SCHED_H
#define VSNRAY_DETAIL_TBB_SCHED_H 1

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

#include "basic_tiled_sched.h"
#include "range.h"

namespace visionaray
{

struct tbb_sched_backend
{
    explicit tbb_sched_backend(unsigned num_threads)
        : init_(num_threads)
    {
    }

    void reset(unsigned num_threads)
    {
        init_.initialize(num_threads);
    }

    template <typename Func>
    void for_each_packet(
            tiled_range2d<int> const& tr,
            int packet_width,
            int packet_height,
            Func const& func
            )
    {
        int x0 = tr.rows().begin();
        int y0 = tr.cols().begin();

        int dx = tr.rows().tile_size();
        int dy = tr.cols().tile_size();

        int nx = tr.rows().end();
        int ny = tr.cols().end();

        tbb::parallel_for(
            tbb::blocked_range2d<int>(x0, nx, dx, y0, ny, dy),
            [=](tbb::blocked_range2d<int> const& r)
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

    tbb::task_scheduler_init init_;
};

template <typename R>
using tbb_sched = basic_tiled_sched<tbb_sched_backend, R>;

} // visionaray

#endif // VSNRAY_DETAIL_TBB_SCHED_H
