// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <type_traits>
#include <utility>

#include <cuda_runtime_api.h>

#include <visionaray/math/detail/math.h> // div_up
#include <visionaray/make_generator.h>

#include "../make_random_seed.h"
#include "../packet_traits.h"
#include "sched_common.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// CUDA kernels
//

template <
    typename R,
    typename PxSamplerT,
    typename Rect,
    typename RTRef,
    typename K,
    typename ...Args
    >
__global__ void render(
        PxSamplerT      sample_params,
        Rect            scissor_box,
        unsigned        frame_id,
        RTRef           rt_ref,
        K               kernel,
        Args...         args
        )
{
    using S = typename R::scalar_type;
    using I = typename simd::int_type<S>::type;

    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < scissor_box.x || y < scissor_box.y || x >= scissor_box.w || y >= scissor_box.h)
    {
        return;
    }

    expand_pixel<S> ep;
    auto seed = make_random_seed(
        convert_to_int(ep.y(y)) * rt_ref.width() + convert_to_int(ep.x(x)),
        I(frame_id)
        );

    auto gen = make_generator(S{}, sample_params, seed);

    auto r = detail::make_primary_rays(
            R{},
            sample_params,
            gen,
            x,
            y,
            args...
            );

    sample_pixel(
            kernel,
            sample_params,
            r,
            gen,
            rt_ref,
            x,
            y,
            args...
            );
}

template <
    typename R,
    typename PxSamplerT,
    typename Intersector,
    typename Rect,
    typename RTRef,
    typename K,
    typename ...Args
    >
__global__ void render(
        detail::have_intersector_tag    /* */,
        PxSamplerT                      sample_params,
        Intersector                     intersector,
        Rect                            scissor_box,
        unsigned                        frame_id,
        RTRef                           rt_ref,
        K                               kernel,
        Args...                         args
        )
{
    using S = typename R::scalar_type;
    using I = typename simd::int_type<S>::type;

    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < scissor_box.x || y < scissor_box.y || x >= scissor_box.w || y >= scissor_box.h)
    {
        return;
    }

    expand_pixel<S> ep;
    auto seed = make_random_seed(
        convert_to_int(ep.y(y)) * rt_ref.width() + convert_to_int(ep.x(x)),
        I(frame_id)
        );

    // TODO: support any sampler
    random_generator<typename R::scalar_type> gen(seed);

    auto r = detail::make_primary_rays(
            R{},
            sample_params,
            gen,
            x,
            y,
            args...
            );

    sample_pixel(
            detail::have_intersector_tag(),
            intersector,
            kernel,
            sample_params,
            r,
            gen,
            rt_ref,
            x,
            y,
            args...
            );
}


//-------------------------------------------------------------------------------------------------
// Dispatch functions
//

template <typename R, typename SP, typename Rect, typename ...Args>
inline void cuda_sched_impl_call_render(
        std::false_type     /* has intersector */,
        SP&                 sparams,
        unsigned            frame_id,
        dim3 const&         grid_size,
        dim3 const&         block_size,
        size_t              smem,
        cudaStream_t const& stream,
        Rect const&         scissor_box,
        Args&&...           args
        )
{
    render<R, typename SP::pixel_sampler_type><<<grid_size, block_size, smem, stream>>>(
            sparams.sample_params,
            scissor_box,
            frame_id,
            std::forward<Args>(args)...
            );
}

template <typename R, typename SP, typename Rect, typename ...Args>
inline void cuda_sched_impl_call_render(
        std::true_type      /* has intersector */,
        SP const&           sparams,
        unsigned            frame_id,
        dim3 const&         grid_size,
        dim3 const&         block_size,
        size_t              smem,
        cudaStream_t const& stream,
        Rect const&         scissor_box,
        Args&&...           args
        )
{
    render<R, typename SP::pixel_sampler_type><<<grid_size, block_size, smem, stream>>>(
            detail::have_intersector_tag(),
            sparams.sample_params,
            sparams.intersector,
            scissor_box,
            frame_id,
            std::forward<Args>(args)...
            );
}


template <typename R, typename K, typename SP>
inline void cuda_sched_impl_frame(
        K                   kernel,
        SP                  sparams,
        unsigned            frame_id,
        dim3 const&         block_size,
        size_t              smem,
        cudaStream_t const& stream
        )
{
    using cuda_dim_t = decltype(block_size.x);

    auto w = static_cast<cuda_dim_t>(sparams.rt.width());
    auto h = static_cast<cuda_dim_t>(sparams.rt.height());

    dim3 grid_size(
            div_up(w, block_size.x),
            div_up(h, block_size.y)
            );

    cuda_sched_impl_call_render<R>(
            typename detail::sched_params_has_intersector<SP>::type(),
            sparams,
            frame_id,
            grid_size,
            block_size,
            smem,
            stream,
            sparams.scissor_box,
            sparams.rt.ref(),
            kernel,
            sparams.rt.width(),
            sparams.rt.height(),
            sparams.cam
            );
}

} // detail


//-------------------------------------------------------------------------------------------------
// cuda_sched implementation
//

template <typename R>
cuda_sched<R>::cuda_sched(vec2ui block_size)
    : block_size_(block_size)
{
}

template <typename R>
cuda_sched<R>::cuda_sched(unsigned block_size_x, unsigned block_size_y)
    : block_size_(block_size_x, block_size_y)
{
}

template <typename R>
template <typename K, typename SP>
void cuda_sched<R>::frame(K kernel, SP sched_params, size_t smem, cudaStream_t const& stream)
{
    sched_params.cam.begin_frame();

    sched_params.rt.begin_frame();

    detail::cuda_sched_impl_frame<R>(
            kernel,
            sched_params,
            frame_id_,
            dim3(block_size_.x, block_size_.y),
            smem,
            stream
            );

    sched_params.rt.end_frame();

    sched_params.cam.end_frame();

    ++frame_id_;
}

} // visionaray
