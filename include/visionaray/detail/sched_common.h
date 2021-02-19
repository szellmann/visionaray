// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_SCHED_COMMON_H
#define VSNRAY_DETAIL_SCHED_COMMON_H 1

#include <utility>

#include <visionaray/math/forward.h>
#include <visionaray/math/matrix.h>
#include <visionaray/math/vector.h>
#include <visionaray/packet_traits.h>
#include <visionaray/pixel_format.h>
#include <visionaray/pixel_sampler_types.h>
#include <visionaray/render_target.h>
#include <visionaray/result_record.h>

#include "macros.h"
#include "pixel_access.h"
#include "tags.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Invoke kernel
//

template <typename K, typename R, typename Generator>
VSNRAY_FUNC
inline auto invoke_kernel(K kernel, R r, Generator& gen, int x, int y)
    -> decltype(kernel(r))
{
    VSNRAY_UNUSED(gen);
    VSNRAY_UNUSED(x);
    VSNRAY_UNUSED(y);

    return kernel(r);
}

template <
    typename K,
    typename R,
    typename Generator,
    typename = void
    >
VSNRAY_FUNC
inline auto invoke_kernel(K kernel, R r, Generator& gen, int x, int y)
    -> decltype(kernel(r, gen))
{
    VSNRAY_UNUSED(x);
    VSNRAY_UNUSED(y);

    return kernel(r, gen);
}

template <
    typename K,
    typename R,
    typename Generator,
    typename = void,
    typename = void
    >
VSNRAY_FUNC
inline auto invoke_kernel(K kernel, R r, Generator& gen, int x, int y)
    -> decltype(kernel(r, x, y))
{
    VSNRAY_UNUSED(gen);

    return kernel(r, x, y);
}

template <
    typename K,
    typename R,
    typename Generator,
    typename = void,
    typename = void,
    typename = void
    >
VSNRAY_FUNC
inline auto invoke_kernel(K kernel, R r, Generator& gen, int x, int y)
    -> decltype(kernel(r, gen, x, y))
{
    return kernel(r, gen, x, y);
}


//-------------------------------------------------------------------------------------------------
// Invoke cam::primary_ray()
//
// Issues the right call based on whether primary_ray() requires or does not require
// a random generator
//

template <typename R, typename Camera, typename Generator, typename T>
VSNRAY_FUNC
inline auto invoke_cam_primary_ray(
        R             /* */,
        Camera const& cam,
        Generator&    gen,
        T const&      x,
        T const&      y,
        T const&      width,
        T const&      height
        )
    -> decltype(cam.primary_ray(R{}, x, y, width, height))
{
    VSNRAY_UNUSED(gen);

    return cam.primary_ray(R{}, x, y, width, height);
}

template <typename R, typename Camera, typename Generator, typename T, typename = void>
VSNRAY_FUNC
inline auto invoke_cam_primary_ray(
        R             /* */,
        Camera const& cam,
        Generator&    gen,
        T const&      x,
        T const&      y,
        T const&      width,
        T const&      height
        )
    -> decltype(cam.primary_ray(R{}, gen, x, y, width, height))
{
    return cam.primary_ray(R{}, gen, x, y, width, height);
}


//-------------------------------------------------------------------------------------------------
// make primary rays
//
// Generates a single (or several when using anti-aliased rendering) primary rays
// for a pixel position
//

template <typename R, typename Generator, typename Camera>
VSNRAY_FUNC
inline R make_primary_rays(
        R                           /* */,
        pixel_sampler::uniform_type /* */,
        Generator&                  gen,
        int                         x,
        int                         y,
        int                         width,
        int                         height,
        Camera const&               cam
        )
{
    using T = typename R::scalar_type;

    return invoke_cam_primary_ray(
            R{},
            cam,
            gen,
            expand_pixel<T>().x(x),
            expand_pixel<T>().y(y),
            T(width),
            T(height)
            );
}

template <typename R, typename T, typename Generator, typename Camera>
VSNRAY_FUNC
inline R make_primary_rays(
        R                               /* */,
        pixel_sampler::basic_jittered_blend_type<T> /* */,
        Generator&                                  gen,
        int                                         x,
        int                                         y,
        int                                         width,
        int                                         height,
        Camera const&                               cam
        )
{
    using U = typename R::scalar_type;

    vector<2, U> jitter(gen.next() - U(0.5), gen.next() - U(0.5));

    return invoke_cam_primary_ray(
            R{},
            cam,
            gen,
            expand_pixel<U>().x(x) + jitter.x,
            expand_pixel<U>().y(y) + jitter.y,
            U(width),
            U(height)
            );
}


//-------------------------------------------------------------------------------------------------
// Depth transform
//

template <typename T, typename Camera>
VSNRAY_FUNC
inline T depth_transform(vector<3, T> const& isect_pos, Camera cam)
{
    matrix<4, 4, T> view_matrix(cam.get_view_matrix());
    matrix<4, 4, T> proj_matrix(cam.get_proj_matrix());

    auto pos4 = proj_matrix * (view_matrix * vector<4, T>(isect_pos, T(1.0)));
    auto pos3 = pos4.xyz() / pos4.w;

    return (pos3.z + T(1.0)) * T(0.5);
}


//-------------------------------------------------------------------------------------------------
// Simple uniform pixel sampler
//

template <
    typename K,
    typename R,
    typename Generator,
    pixel_format CF,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                               kernel,
        pixel_sampler::uniform_type     /* */,
        R const&                        r,
        Generator&                      gen,
        render_target_ref<CF>           rt_ref,
        int                             x,
        int                             y,
        int                             width,
        int                             height,
        Camera const&                   cam
        )
{
    VSNRAY_UNUSED(cam);

    auto result = invoke_kernel(kernel, r, gen, x, y);
    pixel_access::store(
            pixel_format_constant<CF>{},
            pixel_format_constant<PF_RGBA32F>{},
            x,
            y,
            width,
            height,
            result,
            rt_ref.color()
            );
}

template <
    typename K,
    typename R,
    typename Generator,
    pixel_format CF,
    pixel_format DF,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                               kernel,
        pixel_sampler::uniform_type     /* */,
        R const&                        r,
        Generator&                      gen,
        render_target_ref<CF, DF>       rt_ref,
        int                             x,
        int                             y,
        int                             width,
        int                             height,
        Camera const&                   cam
        )
{
    auto result = invoke_kernel(kernel, r, gen, x, y);
    result.depth = select( result.hit, depth_transform(result.isect_pos, cam), typename R::scalar_type(1.0) );
    pixel_access::store(
            pixel_format_constant<CF>{},
            pixel_format_constant<PF_RGBA32F>{},
            pixel_format_constant<DF>{},
            pixel_format_constant<PF_DEPTH32F>{},
            x,
            y,
            width,
            height,
            result,
            rt_ref.color(),
            rt_ref.depth()
            );
}


//-------------------------------------------------------------------------------------------------
// Jittered pixel sampler, result is blended on top of color buffer
//

template <
    typename K,
    typename T,
    typename R,
    typename Generator,
    pixel_format CF,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                                           kernel,
        pixel_sampler::basic_jittered_blend_type<T> blend_params,
        R const&                                    r,
        Generator&                                  gen,
        render_target_ref<CF>                       rt_ref,
        int                                         x,
        int                                         y,
        int                                         width,
        int                                         height,
        Camera const&                               cam
        )
{
    VSNRAY_UNUSED(cam);

    auto result = invoke_kernel(kernel, r, gen, x, y);

    pixel_access::blend(
            pixel_format_constant<CF>{},
            pixel_format_constant<PF_RGBA32F>{},
            x,
            y,
            width,
            height,
            result,
            rt_ref.color(),
            blend_params.sfactor,
            blend_params.dfactor
            );
}

template <
    typename K,
    typename T,
    typename R,
    typename Generator,
    pixel_format CF,
    pixel_format DF,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                                           kernel,
        pixel_sampler::basic_jittered_blend_type<T> blend_params,
        R const&                                    r,
        Generator&                                  gen,
        render_target_ref<CF, DF>                   rt_ref,
        int                                         x,
        int                                         y,
        int                                         width,
        int                                         height,
        Camera const&                               cam
        )
{
    using S = typename R::scalar_type;

    auto result = invoke_kernel(kernel, r, gen, x, y);

    result.depth = select( result.hit, depth_transform(result.isect_pos, cam), S(1.0) );

    pixel_access::blend(
            pixel_format_constant<CF>{},
            pixel_format_constant<PF_RGBA32F>{},
            pixel_format_constant<DF>{},
            pixel_format_constant<PF_DEPTH32F>{},
            x,
            y,
            width,
            height,
            result,
            rt_ref.color(),
            rt_ref.depth(),
            blend_params.sfactor,
            blend_params.dfactor
            );
}

//-------------------------------------------------------------------------------------------------
// w/o intersector
//

template <typename ...Args>
VSNRAY_FUNC
inline void sample_pixel_choose_intersector_impl(Args&&... args)
{
    detail::sample_pixel_impl(std::forward<Args>(args)...);
}


//-------------------------------------------------------------------------------------------------
// w/ intersector
//

template <typename K, typename I>
struct call_kernel_with_intersector
{
    K& kernel;
    I& isect;

    VSNRAY_FUNC
    explicit call_kernel_with_intersector(K& k, I& i) : kernel(k) , isect(i) {}

    template <typename ...Args>
    VSNRAY_FUNC
    auto operator()(Args&&... args) const
        -> decltype( kernel(isect, std::forward<Args>(args)...) )
    {
        return kernel(isect, std::forward<Args>(args)...);
    }
};

template <typename K, typename Intersector, typename ...Args>
VSNRAY_FUNC
inline void sample_pixel_choose_intersector_impl(
        have_intersector_tag    /* */,
        Intersector&            isect,
        K                       kernel,
        Args&&...               args
        )
{
    auto caller = call_kernel_with_intersector<K, Intersector>(kernel, isect);
    detail::sample_pixel_impl(
            caller,
            std::forward<Args>(args)...
            );
}

} // detail


//-------------------------------------------------------------------------------------------------
// Dispatch pixel sampler
//

template <typename ...Args>
VSNRAY_FUNC
inline void sample_pixel(Args&&...  args)
{
    detail::sample_pixel_choose_intersector_impl(std::forward<Args>(args)...);
}

} // visionaray

#endif // VSNRAY_DETAIL_SCHED_COMMON_H
