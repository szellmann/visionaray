// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TBB_SCHED_H
#define VSNRAY_DETAIL_TBB_SCHED_H 1

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

#include "basic_tiled_sched.h"

namespace visionaray
{

struct tbb_sched_backend
{
    using tiled_range_type = tbb::blocked_range2d<int>;
    using range_type = tbb::blocked_range2d<int>;

    explicit tbb_sched_backend(unsigned num_threads)
        : init_(num_threads)
    {
    }

    void reset(unsigned num_threads)
    {
        init_.initialize(num_threads);
    }

    template <typename Func>
    void parallel_for(range_type const& r, Func const& func)
    {
        tbb::parallel_for(r, func);
    }

    tbb::task_scheduler_init init_;
};

template <typename R>
using tbb_sched = basic_tiled_sched<tbb_sched_backend, R>;

} // visionaray

#endif // VSNRAY_DETAIL_TBB_SCHED_H
