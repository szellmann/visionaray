// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_SIMPLE_SCHED_H
#define VSNRAY_DETAIL_SIMPLE_SCHED_H 1

namespace visionaray
{

template <typename R>
class simple_sched
{
public:

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params);

private:

    unsigned frame_id_ = 0;

};

} // visionaray

#include "simple_sched.inl"

#endif // VSNRAY_DETAIL_SIMPLE_SCHED_H
