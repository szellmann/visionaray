// This file is distributed under the MIT license.
// See the LICENSE file for details.
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.

#pragma once

#ifndef VSNRAY_DETAIL_HIP_SCHED_H
#define VSNRAY_DETAIL_HIP_SCHED_H 1

#include <cstddef>

#include <hip/hip_runtime_api.h>

#include <visionaray/math/forward.h>
#include <visionaray/math/vector.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// AMD HIP-based scheduler
//
//
// Uses a 2-D HIP grid to schedule ray traversal kernels
// Default: divide the window into blocks of 16x16 pixels
// Call hip_sched(vec2ui block_dim) for custom block sizes
//
//-------------------------------------------------------------------------------------------------

template <typename R>
class hip_sched
{
public:

    hip_sched() = default;
    hip_sched(vec2ui block_size);
    hip_sched(unsigned block_size_x, unsigned block_size_y);

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params, size_t smem = 0, hipStream_t const& stream = 0);

private:

    vec2ui block_size_ = vec2ui(16, 16);
    unsigned frame_id_ = 0;

};

} // visionaray

#include "hip_sched.inl"

#endif // VSNRAY_DETAIL_HIP_SCHED_H
