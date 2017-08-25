// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_CUDA_SCHED_H
#define VSNRAY_DETAIL_CUDA_SCHED_H 1

#include <visionaray/math/forward.h>
#include <visionaray/math/vector.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// NVIDIA CUDA-based scheduler
//
//
// Uses a 2-D CUDA grid to schedule ray traversal kernels
// Default: divide the window into blocks of 16x16 pixels
// Call cuda_sched(vec2ui block_dim) for custom block sizes
//
//-------------------------------------------------------------------------------------------------

template <typename R>
class cuda_sched
{
public:

    cuda_sched() = default;
    cuda_sched(vec2ui block_size);
    cuda_sched(unsigned block_size_x, unsigned block_size_y);

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params, unsigned frame_num = 0);

private:

    vec2ui block_size_ = vec2ui(16, 16);

};

} // visionaray

#include "cuda_sched.inl"

#endif // VSNRAY_DETAIL_CUDA_SCHED_H
