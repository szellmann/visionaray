// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TBB_SCHED_H
#define VSNRAY_DETAIL_TBB_SCHED_H 1

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#if 1 // TODO: find out when that API changed
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <tbb/global_control.h>
#else
#include <tbb/task_scheduler_init.h>
#endif

#include "basic_sched.h"
#include "range.h"

namespace visionaray
{

struct tbb_sched_backend
{
    explicit tbb_sched_backend(unsigned num_threads)
#if 1 // TODO: find out when that API changed
        : tbb_gc_(new tbb::global_control(tbb::global_control::max_allowed_parallelism, num_threads))
#else
        : init_(num_threads)
#endif
    {
    }

    void reset(unsigned num_threads)
    {
#if 1 // TODO: find out when that API changed
        tbb_gc_.reset(new tbb::global_control(tbb::global_control::max_allowed_parallelism, num_threads));
#else
        init_.initialize(num_threads);
#endif
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

#if 1 // TODO: find out when that API changed
    std::unique_ptr<tbb::global_control> tbb_gc_;
#else
    tbb::task_scheduler_init init_;
#endif
};

template <typename R>
using tbb_sched = basic_sched<tbb_sched_backend, R>;

} // visionaray

#endif // VSNRAY_DETAIL_TBB_SCHED_H
