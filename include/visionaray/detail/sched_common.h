// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_SCHED_COMMON_H
#define VSNRAY_DETAIL_SCHED_COMMON_H 1

#include <chrono>
#include <utility>

#include <visionaray/array.h>
#include <visionaray/matrix_camera.h>
#include <visionaray/packet_traits.h>
#include <visionaray/pixel_format.h>
#include <visionaray/render_target.h>
#include <visionaray/result_record.h>
#include <visionaray/tags.h>

#include "macros.h"
#include "pixel_access.h"
#include "tags.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// TODO: move to a better place
//

#if defined(__CUDA_ARCH__)
VSNRAY_GPU_FUNC
inline unsigned tic()
{
    return clock64();
}
#else
inline unsigned tic()
{
    auto t = std::chrono::high_resolution_clock::now();
    return t.time_since_epoch().count();
}
#endif


//-------------------------------------------------------------------------------------------------
// Invoke kernel
//

template <typename K, typename R, typename Sampler>
VSNRAY_FUNC
inline auto invoke_kernel(K kernel, R r, Sampler& samp, unsigned x, unsigned y)
    -> decltype(kernel(r))
{
    VSNRAY_UNUSED(samp);
    VSNRAY_UNUSED(x);
    VSNRAY_UNUSED(y);

    return kernel(r);
}

template <
    typename K,
    typename R,
    typename Sampler,
    typename = void
    >
VSNRAY_FUNC
inline auto invoke_kernel(K kernel, R r, Sampler& samp, unsigned x, unsigned y)
    -> decltype(kernel(r, samp))
{
    VSNRAY_UNUSED(x);
    VSNRAY_UNUSED(y);

    return kernel(r, samp);
}

template <
    typename K,
    typename R,
    typename Sampler,
    typename = void,
    typename = void
    >
VSNRAY_FUNC
inline auto invoke_kernel(K kernel, R r, Sampler& samp, unsigned x, unsigned y)
    -> decltype(kernel(r, x, y))
{
    VSNRAY_UNUSED(samp);

    return kernel(r, x, y);
}


//-------------------------------------------------------------------------------------------------
// make primary rays
//
// Generates a single (or several when using anti-aliased rendering) primary rays
// for a pixel position
//

template <typename R, typename T, typename Camera>
VSNRAY_FUNC
inline R make_primary_ray_impl(
        T             x,
        T             y,
        size_t        width,
        size_t        height,
        Camera const& cam
        )
{
    return cam.primary_ray(
            R{},
            x,
            y,
            T(width),
            T(height)
            );
}

template <typename R, typename Sampler, typename ...Args>
VSNRAY_FUNC
inline R make_primary_rays(
        R                           /* */,
        pixel_sampler::uniform_type /* */,
        Sampler&                    samp,
        unsigned                    x,
        unsigned                    y,
        Args&&...                   args
        )
{
    VSNRAY_UNUSED(samp);

    using S = typename R::scalar_type;
    return make_primary_ray_impl<R>(expand_pixel<S>().x(x), expand_pixel<S>().y(y), std::forward<Args>(args)...);
}

template <typename R, typename Sampler, typename ...Args>
VSNRAY_FUNC
inline R make_primary_rays(
        R                               /* */,
        pixel_sampler::jittered_type    /* */,
        Sampler&                        samp,
        unsigned                        x,
        unsigned                        y,
        Args&&...                       args
        )
{
    using S = typename R::scalar_type;

    vector<2, S> jitter( samp.next() - S(0.5), samp.next() - S(0.5) );

    return make_primary_ray_impl<R>(
            expand_pixel<S>().x(x) + jitter.x,
            expand_pixel<S>().y(y) + jitter.y,
            std::forward<Args>(args)...
            );
}


// 2x SSAA ------------------------------------------------

template <typename R, typename Sampler, typename ...Args>
VSNRAY_FUNC
inline array<R, 2> make_primary_rays(
        R                           /* */,
        pixel_sampler::ssaa_type<2> /* */,
        Sampler&                    samp,
        unsigned                    x,
        unsigned                    y,
        Args&&...                   args
        )
{
    VSNRAY_UNUSED(samp);

    using S = typename R::scalar_type;

    return {{
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) - S(0.25), expand_pixel<S>().y(y) - S(0.25), std::forward<Args>(args)...),
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) + S(0.25), expand_pixel<S>().y(y) + S(0.25), std::forward<Args>(args)...),
        }};
}

// 4x SSAA ------------------------------------------------

template <typename R, typename Sampler, typename ...Args>
VSNRAY_FUNC
inline array<R, 4> make_primary_rays(
        R                           /* */,
        pixel_sampler::ssaa_type<4> /* */,
        Sampler&                    samp,
        unsigned                    x,
        unsigned                    y,
        Args&&...                   args
        )
{
    VSNRAY_UNUSED(samp);

    using S = typename R::scalar_type;

    return {{
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) - S(0.125), expand_pixel<S>().y(y) - S(0.375), std::forward<Args>(args)...),
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) + S(0.375), expand_pixel<S>().y(y) - S(0.125), std::forward<Args>(args)...),
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) + S(0.125), expand_pixel<S>().y(y) + S(0.375), std::forward<Args>(args)...),
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) - S(0.375), expand_pixel<S>().y(y) + S(0.125), std::forward<Args>(args)...)
        }};
}

// 8x SSAA ------------------------------------------------

template <typename R, typename Sampler, typename ...Args>
VSNRAY_FUNC
inline array<R, 8> make_primary_rays(
        R                           /* */,
        pixel_sampler::ssaa_type<8> /* */,
        Sampler&                    samp,
        unsigned                    x,
        unsigned                    y,
        Args&&...                   args
        )
{
    VSNRAY_UNUSED(samp);

    using S = typename R::scalar_type;

    return {{
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) - S(0.125), expand_pixel<S>().y(y) - S(0.4375), std::forward<Args>(args)...),
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) + S(0.375), expand_pixel<S>().y(y) - S(0.3125), std::forward<Args>(args)...),
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) - S(0.375), expand_pixel<S>().y(y) - S(0.1875), std::forward<Args>(args)...),
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) + S(0.125), expand_pixel<S>().y(y) - S(0.0625), std::forward<Args>(args)...),
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) - S(0.125), expand_pixel<S>().y(y) + S(0.0625), std::forward<Args>(args)...),
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) + S(0.375), expand_pixel<S>().y(y) + S(0.1825), std::forward<Args>(args)...),
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) - S(0.375), expand_pixel<S>().y(y) + S(0.3125), std::forward<Args>(args)...),
        make_primary_ray_impl<R>(expand_pixel<S>().x(x) + S(0.125), expand_pixel<S>().y(y) + S(0.4375), std::forward<Args>(args)...)
        }};
}


template <
    size_t   Num,
    typename R,
    typename PxSamplerT,
    typename Sampler,
    typename ...Args
    >
VSNRAY_FUNC
inline array<R, Num> make_primary_rays(
        R               /* */,
        PxSamplerT      /* */,
        Sampler&        samp,
        unsigned        x,
        unsigned        y,
        Args&&...       args
        )
{
    array<R, Num> result;

    for (size_t i = 0; i < Num; ++i)
    {
        result[i] = make_primary_rays(
                R{},
                PxSamplerT{},
                samp,
                x,
                y,
                std::forward<Args>(args)...
                );
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Depth transform
//

template <typename T>
VSNRAY_FUNC
inline T depth_transform(vector<3, T> const& isect_pos, matrix_camera const& cam)
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
    typename Sampler,
    pixel_format CF,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                               kernel,
        pixel_sampler::uniform_type     /* */,
        R const&                        r,
        Sampler&                        samp,
        unsigned                        frame_num,
        render_target_ref<CF>           rt_ref,
        int                             x,
        int                             y,
        int                             width,
        int                             height,
        Camera const&                   cam
        )
{
    VSNRAY_UNUSED(frame_num);
    VSNRAY_UNUSED(cam);

    auto result = invoke_kernel(kernel, r, samp, x, y);
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
    typename Sampler,
    pixel_format CF,
    pixel_format DF,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                               kernel,
        pixel_sampler::uniform_type     /* */,
        R const&                        r,
        Sampler&                        samp,
        unsigned                        frame_num,
        render_target_ref<CF, DF>       rt_ref,
        int                             x,
        int                             y,
        int                             width,
        int                             height,
        Camera const&                   cam
        )
{
    VSNRAY_UNUSED(frame_num);

    auto result = invoke_kernel(kernel, r, samp, x, y);
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
// Jittered pixel sampler, sampler is passed to kernel
//

template <
    typename K,
    typename R,
    typename Sampler,
    pixel_format CF,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                               kernel,
        pixel_sampler::jittered_type    /* */,
        R const&                        r,
        Sampler&                        samp,
        unsigned                        frame_num,
        render_target_ref<CF>           rt_ref,
        int                             x,
        int                             y,
        int                             width,
        int                             height,
        Camera const&                   cam
        )
{
    VSNRAY_UNUSED(frame_num);
    VSNRAY_UNUSED(cam);

    auto result = invoke_kernel(kernel, r, samp, x, y);
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


//-------------------------------------------------------------------------------------------------
// Jittered pixel sampler, result is blended on top of color buffer, sampler is passed to kernel
//

template <
    typename K,
    typename R,
    typename Sampler,
    pixel_format CF,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                                   kernel,
        pixel_sampler::jittered_blend_type  /* */,
        R const&                            r,
        Sampler&                            samp,
        unsigned                            frame_num,
        render_target_ref<CF>               rt_ref,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        Camera const&                       cam
        )
{
    VSNRAY_UNUSED(cam);

    using S = typename R::scalar_type;

    auto result = invoke_kernel(kernel, r, samp, x, y);
    auto alpha  = S(1.0) / S(frame_num);

    pixel_access::blend(
            pixel_format_constant<CF>{},
            pixel_format_constant<PF_RGBA32F>{},
            x,
            y,
            width,
            height,
            result,
            rt_ref.color(),
            alpha, S(1.0) - alpha
            );
}

template <
    typename K,
    typename R,
    typename Sampler,
    pixel_format CF,
    pixel_format DF,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                                   kernel,
        pixel_sampler::jittered_blend_type  /* */,
        R const&                            r,
        Sampler&                            samp,
        unsigned                            frame_num,
        render_target_ref<CF, DF>           rt_ref,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        Camera const&                       cam
        )
{
    using S = typename R::scalar_type;

    auto result = invoke_kernel(kernel, r, samp, x, y);
    auto alpha  = S(1.0) / S(frame_num);

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
            alpha, S(1.0) - alpha
            );
}


//-------------------------------------------------------------------------------------------------
// jittered pixel sampler, blends several samples at once
//

template <
    typename K,
    typename R,
    size_t Num,
    typename Sampler,
    pixel_format CF,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                                   kernel,
        pixel_sampler::jittered_blend_type  /* */,
        array<R, Num> const&                rays,
        Sampler&                            samp,
        unsigned                            frame_num,
        render_target_ref<CF>               rt_ref,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        Camera const&                       cam
        )
{
    VSNRAY_UNUSED(cam);

    using S = typename R::scalar_type;

    auto ray_ptr = rays.data();

    auto frame_begin = frame_num;
    auto frame_end   = frame_num + Num;

    for (size_t frame = frame_begin; frame < frame_end; ++frame)
    {
        auto result = invoke_kernel(kernel, *ray_ptr++, samp, x, y);
        auto alpha = S(1.0) / S(frame);
        pixel_access::blend(
                pixel_format_constant<CF>{},
                pixel_format_constant<PF_RGBA32F>{},
                x,
                y,
                width,
                height,
                result,
                rt_ref.color(),
                alpha, S(1.0) - alpha
                );
    }
}


//-------------------------------------------------------------------------------------------------
// SSAA pixel sampler
//

template <
    typename K,
    typename R,
    size_t Num,
    typename Sampler,
    pixel_format CF,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                               kernel,
        pixel_sampler::ssaa_type<Num>   /* */,
        array<R, Num> const&            rays,
        Sampler&                        samp,
        unsigned                        frame_num,
        render_target_ref<CF>           rt_ref,
        int                             x,
        int                             y,
        int                             width,
        int                             height,
        Camera const&                   cam
        )
{
    VSNRAY_UNUSED(frame_num);
    VSNRAY_UNUSED(cam);

    using S     = typename R::scalar_type;
    using Color = typename decltype(invoke_kernel(kernel, R{}, samp, x, y))::color_type;

    auto ray_ptr = rays.data();

    auto frame_begin = 0;
    auto frame_end   = Num;

    pixel_access::store(
            pixel_format_constant<CF>{},
            pixel_format_constant<PF_RGBA32F>{},
            x,
            y,
            width,
            height,
            Color(0.0),
            rt_ref.color()
            );

    for (size_t frame = frame_begin; frame < frame_end; ++frame)
    {
        auto result = invoke_kernel(kernel, *ray_ptr++, samp, x, y);
        auto alpha = S(1.0) / S(Num);
        pixel_access::blend(
                pixel_format_constant<CF>{},
                pixel_format_constant<PF_RGBA32F>{},
                x,
                y,
                width,
                height,
                result,
                rt_ref.color(),
                alpha,
                S(1.0)
                );
    }
}

template <
    typename K,
    typename R,
    size_t Num,
    typename Sampler,
    pixel_format CF,
    pixel_format DF,
    typename Camera
    >
VSNRAY_FUNC
inline void sample_pixel_impl(
        K                               kernel,
        pixel_sampler::ssaa_type<Num>   /* */,
        array<R, Num> const&            rays,
        Sampler&                        samp,
        unsigned                        frame_num,
        render_target_ref<CF, DF>       rt_ref,
        int                             x,
        int                             y,
        int                             width,
        int                             height,
        Camera const&                   cam
        )
{
    using S      = typename R::scalar_type;
    using Result = decltype(invoke_kernel(kernel, R{}, samp, x, y));

    auto ray_ptr = rays.data();

    auto frame_begin = frame_num;
    auto frame_end   = frame_num + Num;

    pixel_access::store(
            pixel_format_constant<CF>{},
            pixel_format_constant<PF_RGBA32F>{},
            pixel_format_constant<DF>{},
            pixel_format_constant<PF_DEPTH32F>{},
            x,
            y,
            width,
            height,
            Result{},
            rt_ref.color(),
            rt_ref.depth()
            );

    for (size_t frame = frame_begin; frame < frame_end; ++frame)
    {
        auto result = invoke_kernel(kernel, *ray_ptr++, samp, x, y);
        result.depth = select( result.hit, depth_transform(result.isect_pos, cam), S(1.0) );
        auto alpha = S(1.0) / S(Num);
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
                alpha,
                S(1.0)
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
