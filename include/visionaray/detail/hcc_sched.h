// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_HCC_SCHED_H
#define VSNRAY_DETAIL_HCC_SCHED_H 1

#include <visionaray/math/forward.h>
#include <visionaray/math/vector.h>

namespace visionaray
{

template <typename R>
class hcc_sched
{
public:

    hcc_sched() = default;
    hcc_sched(vec2ui block_size);
    hcc_sched(unsigned block_size_x, unsigned block_size_y);

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params);

private:

    vec2ui block_size_ = vec2ui(16, 16);

};

} // visionaray

#include "hcc_sched.inl"

#endif // VSNRAY_DETAIL_HCC_SCHED_H
