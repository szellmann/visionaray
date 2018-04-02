// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BASIC_TILED_SCHED_H
#define VSNRAY_DETAIL_BASIC_TILED_SCHED_H 1

namespace visionaray
{

template <typename Backend, typename R>
class basic_tiled_sched
{
public:

    explicit basic_tiled_sched(unsigned num_threads);

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params, unsigned frame_num = 0);

    void reset(unsigned num_threads);

private:

    Backend backend_;

};

} // visionaray

#include "basic_tiled_sched.inl"

#endif // VSNRAY_DETAIL_BASIC_TILED_SCHED_H
