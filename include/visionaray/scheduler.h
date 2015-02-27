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

template <typename... Args>
struct sched_params;

template <typename RT, typename PxSamplerT>
struct sched_params<RT, PxSamplerT>
{
    typedef RT                          rt_type;
    typedef typename RT::color_traits   color_traits;
    typedef PxSamplerT                  pixel_sampler_type;

    camera const& cam;
    RT& rt;
};

template <typename MT, typename V, typename RT, typename PxSamplerT>
struct sched_params<MT, V, RT, PxSamplerT>
{
    using rt_type               = RT;
    using color_traits          = typename RT::color_traits;
    using pixel_sampler_type    = PxSamplerT;

    MT const& view_matrix;
    MT const& proj_matrix;
    V const& viewport;
    RT& rt;
};


//-------------------------------------------------------------------------------------------------
// Sched params factory
//

template <typename PxSamplerT, typename RT>
sched_params<RT, PxSamplerT> make_sched_params(camera const& cam, RT& rt)
{
    return sched_params<RT, PxSamplerT>{ cam, rt };
}

template <typename PxSamplerT, typename MT, typename V, typename RT>
sched_params<MT, V, RT, PxSamplerT> make_sched_params(MT const& view_matrix, MT const& proj_matrix,
    V const& viewport, RT& rt)
{
    return sched_params<MT, V, RT, PxSamplerT>{ view_matrix, proj_matrix, viewport, rt };
}

} // visionaray

#ifdef __CUDACC__
#include "detail/cuda_sched.h"
#endif
#include "detail/simple_sched.h"
#include "detail/tiled_sched.h"

#endif // VSNRAY_SCHED_H
