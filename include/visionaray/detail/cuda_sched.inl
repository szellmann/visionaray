// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <type_traits>

#include <cuda_runtime_api.h>

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
    typename Viewport,
    typename RTRef,
    typename K,
    typename ...Args
    >
__global__ void render(
        Viewport        viewport,
        RTRef           rt_ref,
        K               kernel,
        unsigned        frame_num,
        Args...         args
        )
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= viewport.w || y >= viewport.h)
    {
        return;
    }

    sample_pixel<R>(
            kernel,
            PxSamplerT(),
            rt_ref,
            x,
            y,
            frame_num,
            viewport,
            args...
            );
}

template <
    typename R,
    typename PxSamplerT,
    typename Intersector,
    typename Viewport,
    typename RTRef,
    typename K,
    typename ...Args
    >
__global__ void render(
        detail::have_intersector_tag    /* */,
        Intersector                     intersector,
        Viewport                        viewport,
        RTRef                           rt_ref,
        K                               kernel,
        unsigned                        frame_num,
        Args...                         args
        )
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= viewport.w || y >= viewport.h)
    {
        return;
    }

    sample_pixel<R>(
            kernel,
            detail::have_intersector_tag(),
            intersector,
            PxSamplerT(),
            rt_ref,
            x,
            y,
            frame_num,
            viewport,
            args...
            );
}


//-------------------------------------------------------------------------------------------------
// Dispatch functions
//

template <typename R, typename SP, typename Viewport, typename ...Args>
inline void cuda_sched_impl_call_render(
        std::false_type /* has intersector */,
        SP              /* */,
        Viewport const& viewport,
        Args&&...       args
        )
{
    dim3 block_size(16, 16);

    using cuda_dim_t = decltype(block_size.x);

    auto w = static_cast<cuda_dim_t>(viewport.w);
    auto h = static_cast<cuda_dim_t>(viewport.h);

    dim3 grid_size(
            div_up(w, block_size.x),
            div_up(h, block_size.y)
            );

    render<R, typename SP::pixel_sampler_type><<<grid_size, block_size>>>(
            viewport,
            std::forward<Args>(args)...
            );

    cudaPeekAtLastError();
    cudaDeviceSynchronize();
}

template <typename R, typename SP, typename Viewport, typename ...Args>
inline void cuda_sched_impl_call_render(
        std::true_type  /* has intersector */,
        SP const&       sparams,
        Viewport const& viewport,
        Args&&...       args
        )
{
    dim3 block_size(16, 16);

    using cuda_dim_t = decltype(block_size.x);

    auto w = static_cast<cuda_dim_t>(viewport.w);
    auto h = static_cast<cuda_dim_t>(viewport.h);

    dim3 grid_size(
            div_up(w, block_size.x),
            div_up(h, block_size.y)
            );

    render<R, typename SP::pixel_sampler_type><<<grid_size, block_size>>>(
            detail::have_intersector_tag(),
            sparams.intersector,
            viewport,
            std::forward<Args>(args)...
            );

    cudaPeekAtLastError();
    cudaDeviceSynchronize();
}


template <typename R, typename K, typename SP>
inline void cuda_sched_impl_frame(
        std::true_type  /* has matrix */,
        K               kernel,
        SP              sparams,
        unsigned        frame_num
        )
{
    auto rt_ref             = sparams.rt.ref();
    auto inv_view_matrix    = inverse(sparams.view_matrix);
    auto inv_proj_matrix    = inverse(sparams.proj_matrix);
    auto viewport           = sparams.viewport;

    cuda_sched_impl_call_render<R>(
            typename detail::sched_params_has_intersector<SP>::type(),
            sparams,
            viewport,
            rt_ref,
            kernel,
            frame_num,
            sparams.view_matrix,
            inv_view_matrix,
            sparams.proj_matrix,
            inv_proj_matrix
            );
}

template <typename R, typename K, typename SP>
inline void cuda_sched_impl_frame(
        std::false_type /* has matrix */,
        K               kernel,
        SP              sparams,
        unsigned        frame_num
        )
{
    auto rt_ref             = sparams.rt.ref();
    auto viewport           = sparams.cam.get_viewport();

    //  front, side, and up vectors form an orthonormal basis
    auto f = normalize( sparams.cam.eye() - sparams.cam.center() );
    auto s = normalize( cross(sparams.cam.up(), f) );
    auto u =            cross(f, s);

    auto eye   = sparams.cam.eye();
    auto cam_u = s * tan(sparams.cam.fovy() / 2.0f) * sparams.cam.aspect();
    auto cam_v = u * tan(sparams.cam.fovy() / 2.0f);
    auto cam_w = -f;

    cuda_sched_impl_call_render<R>(
            typename detail::sched_params_has_intersector<SP>::type(),
            sparams,
            viewport,
            rt_ref,
            kernel,
            frame_num,
            eye,
            cam_u,
            cam_v,
            cam_w
            );
}

} // detail


template <typename R>
template <typename K, typename SP>
void cuda_sched<R>::frame(K kernel, SP sched_params, unsigned frame_num)
{
    sched_params.rt.begin_frame();

    detail::cuda_sched_impl_frame<R>(
            typename detail::sched_params_has_view_matrix<SP>::type(),
            kernel,
            sched_params,
            frame_num
            );

    sched_params.rt.end_frame();
}

} // visionaray
