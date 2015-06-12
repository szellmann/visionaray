// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SCHED_H
#define VSNRAY_SCHED_H

#include <type_traits>
#include <utility>

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
// Base classes for scheduler params
//

struct sched_params_base {};

template <typename Intersector>
struct sched_params_intersector_base
{
    sched_params_intersector_base(Intersector& i) : intersector(i) {}

    Intersector& intersector;
};


//-------------------------------------------------------------------------------------------------
// Param structs for different pixel sampling strategies
//

template <typename... Args>
struct sched_params;

template <typename Base, typename RT, typename PxSamplerT>
struct sched_params<Base, RT, PxSamplerT> : Base
{
    using has_camera            = std::true_type;
    using has_view_matrix       = std::false_type;
    using has_proj_matrix       = std::false_type;

    using rt_type               = RT;
    using color_traits          = typename RT::color_traits;
    using pixel_sampler_type    = PxSamplerT;

    template <typename ...Args>
    sched_params(camera const& c, RT& r, Args&&... args)
        : Base( std::forward<Args>(args)... )
        , cam(c)
        , rt(r)
    {
    }

    camera const& cam;
    RT& rt;
};

template <typename Base, typename MT, typename V, typename RT, typename PxSamplerT>
struct sched_params<Base, MT, V, RT, PxSamplerT> : Base
{
    using has_camera            = std::false_type;
    using has_view_matrix       = std::true_type;
    using has_proj_matrix       = std::true_type;

    using rt_type               = RT;
    using color_traits          = typename RT::color_traits;
    using pixel_sampler_type    = PxSamplerT;

    template <typename ...Args>
    sched_params(MT const& vm, MT const& pm, V const& vp, RT& r, Args&&... args)
        : Base( std::forward<Args>(args)...)
        , view_matrix(vm)
        , proj_matrix(pm)
        , viewport(vp)
        , rt(r)
    {
    }

    MT const& view_matrix;
    MT const& proj_matrix;
    V const& viewport;
    RT& rt;
};


//-------------------------------------------------------------------------------------------------
// Sched params factory
//

template <typename PxSamplerT, typename RT>
auto make_sched_params(camera const& cam, RT& rt)
    -> sched_params<sched_params_base, RT, PxSamplerT>
{
    return sched_params<sched_params_base, RT, PxSamplerT>{ cam, rt };
}

template <typename PxSamplerT, typename Intersector, typename RT>
auto make_sched_params(camera const& cam, RT& rt, Intersector& isect)
    -> sched_params<sched_params_intersector_base<Intersector>, RT, PxSamplerT>
{
    return sched_params<sched_params_intersector_base<Intersector>, RT, PxSamplerT>{
            cam,
            rt,
            isect
            };
}


template <typename PxSamplerT, typename MT, typename V, typename RT>
auto make_sched_params(
        MT const& view_matrix,
        MT const& proj_matrix,
        V const& viewport,
        RT& rt
        )
    -> sched_params<sched_params_base, MT, V, RT, PxSamplerT>
{
    return sched_params<sched_params_base, MT, V, RT, PxSamplerT>{ view_matrix, proj_matrix, viewport, rt };
}

template <typename PxSamplerT, typename Intersector, typename MT, typename V, typename RT>
auto make_sched_params(
        MT const& view_matrix,
        MT const& proj_matrix,
        V const& viewport,
        RT& rt,
        Intersector& isect
        )
    -> sched_params<sched_params_intersector_base<Intersector>, MT, V, RT, PxSamplerT>
{
    return sched_params<sched_params_intersector_base<Intersector>, MT, V, RT, PxSamplerT>{
            view_matrix,
            proj_matrix,
            viewport,
            rt,
            isect
            };
}

} // visionaray

#ifdef __CUDACC__
#include "detail/cuda_sched.h"
#endif
#include "detail/simple_sched.h"
#if !defined(__MINGW32__) && !defined(__MINGW64__)
#include "detail/tiled_sched.h"
#endif

#endif // VSNRAY_SCHED_H
