// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VIEWER_HOST_DEVICE_SCHED_H
#define VSNRAY_VIEWER_HOST_DEVICE_SCHED_H 1

#include <visionaray/scheduler.h>

#if defined(__INTEL_COMPILER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <visionaray/detail/tbb_sched.h>
#endif

#include "render_state.h"

namespace visionaray
{

class host_device_sched
{
public:

    host_device_sched(render_state const& state);

    template <typename ...Args>
    void frame(Args&&... args);

private:

//  using host_ray_type = basic_ray<float>;
    using host_ray_type = basic_ray<simd::float4>;
//  using host_ray_type = basic_ray<simd::float8>;
//  using host_ray_type = basic_ray<simd::float16>;

    using device_ray_type = basic_ray<float>;

    render_state const& state_;

#if defined(__INTEL_COMPILER) || defined(__MINGW32__) || defined(__MINGW64__)
    tbb_sched<host_ray_type> host_sched_;
#else
    tiled_sched<host_ray_type> host_sched_;
#endif
    cuda_sched<basic_ray<float>> device_sched_;

};

} // visionaray

#include "host_device_sched.inl"

#endif // VSNRAY_VIEWER_HOST_DEVICE_SCHED_H
