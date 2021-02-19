// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_SCHED_COMMON_H
#define VSNRAY_DETAIL_SCHED_COMMON_H 1

#include <type_traits>
#include <utility>

#include <visionaray/math/forward.h>
#include <visionaray/math/matrix.h>
#include <visionaray/math/vector.h>
#include <visionaray/packet_traits.h>
#include <visionaray/matrix_camera.h>
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
// Get SSAA pixel offsets from uniform sampler
//

template <typename T>
VSNRAY_FUNC
inline vector<2, T> get_uniform_pixel_offset(
        T                           /* */,
        pixel_sampler::uniform_type ps,
        unsigned                    s
        )
{
    switch (ps.ssaa_factor)
    {
    case 2: // 2x SSAA
        if (s == 0) return { T(-0.25), T(-0.25) };
        else        return { T( 0.25), T( 0.25) };

    case 4: // 4x SSAA
        if (s == 0)      return { T(-0.125), T(-0.325) };
        else if (s == 1) return { T( 0.375), T(-0.125) };
        else if (s == 2) return { T( 0.125), T( 0.375) };
        else             return { T(-0.375), T( 0.125) };

    case 8: // 8x SSAA
        if (s == 0)      return { T(-0.125), T(-0.4375) };
        else if (s == 1) return { T( 0.375), T(-0.3125) };
        else if (s == 2) return { T(-0.375), T(-0.1875) };
        else if (s == 3) return { T( 0.125), T(-0.0625) };
        else if (s == 4) return { T(-0.125), T( 0.0625) };
        else if (s == 5) return { T( 0.375), T( 0.1825) };
        else if (s == 6) return { T(-0.375), T( 0.1825) };
        else             return { T( 0.125), T( 0.4375) };

    default: // No SSAA
        return { T(0.0), T(0.0) };
    }
}

//-------------------------------------------------------------------------------------------------
// Generates a primary ray for a pixel position
//

template <typename R, typename Generator, typename Camera>
VSNRAY_FUNC
inline R make_primary_ray(
        R                           /* */,
        pixel_sampler::uniform_type ps,
        Generator&                  gen,
        int                         x,
        int                         y,
        int                         width,
        int                         height,
        unsigned                    sample,
        Camera const&               cam
        )
{
    using T = typename R::scalar_type;

    vector<2, T> off = get_uniform_pixel_offset(T{}, ps, sample);

    return invoke_cam_primary_ray(
            R{},
            cam,
            gen,
            expand_pixel<T>().x(x) + off.x,
            expand_pixel<T>().y(y) + off.y,
            T(width),
            T(height)
            );
}

template <typename R, typename T, typename Generator, typename Camera>
VSNRAY_FUNC
inline R make_primary_ray(
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
// Transform depth from ray to OpenGL/raster coordinates
//

template <typename R, typename T, typename Camera>
VSNRAY_FUNC
inline T depth_transform(R, T, Camera)
{
    return T(-1.0f);
}

template <typename R, typename T>
VSNRAY_FUNC
inline T depth_transform(R const& r, T const& depth, matrix_camera const& cam)
{
    vector<3, T> isect_pos = r.ori + r.dir * depth;

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
    typename RenderTargetRef,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                           kernel,
        pixel_sampler::uniform_type ps,
        R                           /* */,
        Generator&                  gen,
        RenderTargetRef             rt_ref,
        int                         x,
        int                         y,
        int                         width,
        int                         height,
        Camera const&               cam
        )
{
    using RR = decltype(invoke_kernel(kernel, R{}, gen, x, y));
    using S = typename RR::scalar_type;

    RR rr;

    for (unsigned s = 0; s < ps.ssaa_factor; ++s)
    {
        auto r = make_primary_ray(
                R{},
                ps,
                gen,
                x,
                y,
                width,
                height,
                s,
                cam
                );

        auto result = invoke_kernel(kernel, r, gen, x, y);

        // Arbitrarily assign the depth of _one_ pixel that recorded a hit
        if (RenderTargetRef::depth_format != PF_UNSPECIFIED)
        {
            result.depth = select(result.hit, depth_transform(r, result.depth, cam), S(1.0));
            rr.depth += result.depth;
        }

        rr.hit |= result.hit;
        rr.color += result.color;
    }

    rr.color /= S((float)ps.ssaa_factor);
    rr.depth /= S((float)ps.ssaa_factor);

    pixel_access::store(
            pixel_format_constant<RenderTargetRef::color_format>{},
            pixel_format_constant<PF_RGBA32F>{},
            x,
            y,
            width,
            height,
            rr.color,
            rt_ref.color()
            );

    if (RenderTargetRef::depth_format != PF_UNSPECIFIED && visionaray::any(rr.hit))
    {
        pixel_access::store(
                pixel_format_constant<RenderTargetRef::depth_format>{},
                pixel_format_constant<PF_DEPTH32F>{},
                x,
                y,
                width,
                height,
                rr.depth,
                rt_ref.depth()
                );
    }
}


//-------------------------------------------------------------------------------------------------
// Jittered pixel sampler, result is blended on top of color buffer
//

template <
    typename K,
    typename T,
    typename R,
    typename Generator,
    typename RenderTargetRef,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                                           kernel,
        pixel_sampler::basic_jittered_blend_type<T> ps,
        R                                           /* */,
        Generator&                                  gen,
        RenderTargetRef                             rt_ref,
        int                                         x,
        int                                         y,
        int                                         width,
        int                                         height,
        Camera const&                               cam
        )
{
    using RR = decltype(invoke_kernel(kernel, R{}, gen, x, y));
    using S = typename RR::scalar_type;

    RR rr;

    for (unsigned s = 0; s < ps.spp; ++s)
    {
        auto r = make_primary_ray(
                R{},
                ps,
                gen,
                x,
                y,
                width,
                height,
                cam
                );

        auto result = invoke_kernel(kernel, r, gen, x, y);

        // Arbitrarily assign the depth of _one_ pixel that recorded a hit
        if (RenderTargetRef::depth_format != PF_UNSPECIFIED)
        {
            result.depth = select(result.hit, depth_transform(r, result.depth, cam), S(1.0));
            rr.depth += result.depth;
        }

        rr.hit |= result.hit;
        rr.color += result.color;
    }

    rr.color /= S((float)ps.spp);
    rr.depth /= S((float)ps.spp);

    pixel_access::blend(
            pixel_format_constant<RenderTargetRef::color_format>{},
            pixel_format_constant<PF_RGBA32F>{},
            x,
            y,
            width,
            height,
            rr.color,
            rt_ref.color(),
            ps.sfactor,
            ps.dfactor
            );

    if (RenderTargetRef::depth_format != PF_UNSPECIFIED && visionaray::any(rr.hit))
    {
        pixel_access::store(
                pixel_format_constant<RenderTargetRef::depth_format>{},
                pixel_format_constant<PF_DEPTH32F>{},
                x,
                y,
                width,
                height,
                rr.depth,
                rt_ref.depth()
                );
    }
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
