// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TILED_SCHED_H
#define VSNRAY_DETAIL_TILED_SCHED_H 1

#include <visionaray/config.h>

#if VSNRAY_HAVE_TBB
#include <tbb/task_scheduler_init.h>
#else
#include "../thread_pool.h"
#endif

namespace visionaray
{

template <typename R>
class tiled_sched
{
public:

    explicit tiled_sched(unsigned num_threads);

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params, unsigned frame_num = 0);

    void reset(unsigned num_threads);

private:

#if VSNRAY_HAVE_TBB
    tbb::task_scheduler_init pool_;
#else
    thread_pool pool_;
#endif

};

} // visionaray

#include "tiled_sched.inl"

#endif // VSNRAY_DETAIL_TILED_SCHED_H
