// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TBB_SCHED_H
#define VSNRAY_DETAIL_TBB_SCHED_H

#include <memory>

#include <visionaray/render_target.h>

namespace visionaray
{

template <typename R>
class tbb_sched
{
public:

    explicit tbb_sched(int num_threads);
   ~tbb_sched();

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params, unsigned frame_num = 0);

private:

    struct impl;
    std::unique_ptr<impl> impl_;

};

} // visionaray

#include "tbb_sched.inl"

#endif // VSNRAY_DETAIL_TBB_SCHED_H
