// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SCHED_H
#define VSNRAY_SCHED_H

#include "camera.h"

namespace visionaray
{

class render_target;

//-------------------------------------------------------------------------------------------------
// Pixel sampler tags for use in scheduler params
//

namespace pixel_sampler
{
struct uniform_type {};
struct jittered_type {};
struct jittered_blend_type {};
}


//-------------------------------------------------------------------------------------------------
// Param structs for different pixel sampling strategies
//

template <typename RT, typename PxSamplerT = pixel_sampler::uniform_type>
struct sched_params
{
    typedef RT                          rt_type;
    typedef typename RT::color_traits   color_traits;
    typedef PxSamplerT                  pixel_sampler_type;

    camera const& cam;
    RT& rt;
};

} // visionaray

#ifdef __CUDACC__
#include "detail/cuda_sched.h"
#endif
#include "detail/simple_sched.h"
#include "detail/tiled_sched.h"

#endif // VSNRAY_SCHED_H


