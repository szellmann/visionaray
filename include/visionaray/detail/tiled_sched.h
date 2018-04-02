// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TILED_SCHED_H
#define VSNRAY_DETAIL_TILED_SCHED_H 1

#include "basic_tiled_sched.h"
#include "parallel_for.h"
#include "thread_pool.h"

namespace visionaray
{

struct tiled_sched_backend
{
    using tiled_range_type = tiled_range2d<int>;
    using range_type = range2d<int>;

    explicit tiled_sched_backend(unsigned num_threads)
        : pool_(num_threads)
    {
    }

    void reset(unsigned num_threads)
    {
        pool_.reset(num_threads);
    }

    template <typename Func>
    void parallel_for(tiled_range2d<int> const& r, Func const& func)
    {
        visionaray::parallel_for(pool_, r, func);
    }

    thread_pool pool_;
};

template <typename R>
using tiled_sched = basic_tiled_sched<tiled_sched_backend, R>;

} // visionaray

#endif // VSNRAY_DETAIL_TILED_SCHED
