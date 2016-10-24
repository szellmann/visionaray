// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_SIMPLE_SCHED_H
#define VSNRAY_DETAIL_SIMPLE_SCHED_H 1

#include <memory>

namespace visionaray
{

template <typename R>
class simple_sched
{
public:

    simple_sched();

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params, unsigned frame_num = 0);

private:

    struct impl;
    std::unique_ptr<impl> const impl_;

};

} // visionaray

#include "simple_sched.inl"

#endif // VSNRAY_DETAIL_SIMPLE_SCHED_H
