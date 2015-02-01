// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_CUDA_SCHED_H
#define VSNRAY_DETAIL_CUDA_SCHED_H

namespace visionaray
{

template <typename R>
class cuda_sched
{
public:

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params, unsigned frame_num);

};

} // visionaray

#include "cuda_sched.inl"

#endif // VSNRAY_DETAIL_CUDA_SCHED_H


