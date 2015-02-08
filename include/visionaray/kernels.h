// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_KERNELS_H
#define VSNRAY_KERNELS_H

#include <visionaray/camera.h>
#include <visionaray/render_target.h>
#include <visionaray/scheduler.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Param structs for use with simple kernel
//

namespace simple
{

template <typename ...Args>
struct kernel_params;

template <typename P, typename N, typename M, typename L, typename ...Args>
struct kernel_params<P, N, M, L, Args...>
{
    typedef P primitive_type;
    typedef N normal_type;
    typedef M material_type;
    typedef L light_type;

    struct
    {
        P begin;
        P end;
    } prims;

    N normals;
    M materials;

    struct
    {
        L begin;
        L end;
    } lights;
};

template <typename P, typename N, typename TC, typename M, typename T, typename L, typename ...Args>
struct kernel_params<P, N, TC, M, T, L, Args...>
{
    typedef P  primitive_type;
    typedef N  normal_type;
    typedef TC tex_coords_type;
    typedef M  material_type;
    typedef T  texture_type;
    typedef L  light_type;

    struct
    {
        P begin;
        P end;
    } prims;

    N  normals;
    TC tex_coords;
    M  materials;
    T  textures;

    struct
    {
        L begin;
        L end;
    } lights;
};


//-------------------------------------------------------------------------------------------------
// Default pixel sampler type for simple kernel
//

typedef pixel_sampler::uniform_type pixel_sampler_type;


//-------------------------------------------------------------------------------------------------
// Factory for param structs
//

template <typename P, typename N, typename M, typename L>
kernel_params<P, N, M, L>  make_params(P const& begin, P const& end, N const& normals,
    M const& materials, L const& lbegin, L const& lend)
{
    return kernel_params<P, N, M, L>
    {
        { begin, end }, normals, materials, { lbegin, lend }
    };
}

template <typename P, typename N, typename TC, typename M, typename T, typename L>
kernel_params<P, N, TC, M, T, L> make_params(P const& begin, P const& end, N const& normals,
    TC const& tex_coords, M const& materials, T const& textures, L const& lbegin, L const& lend)
{
    return kernel_params<P, N, TC, M, T, L>
    {
        { begin, end }, normals, tex_coords, materials, textures, { lbegin, lend }
    };
}

} // simple


//-------------------------------------------------------------------------------------------------
// Param structs for use with whitted kernel
//

namespace whitted
{

using simple::kernel_params;
using simple::make_params;
using pixel_sampler_type = simple::pixel_sampler_type;

} // whitted


//-------------------------------------------------------------------------------------------------
// Param structs for use with path tracing kernel
//

namespace pathtracing
{

using simple::kernel_params;
using simple::make_params;
typedef pixel_sampler::jittered_blend_type pixel_sampler_type;

} // pathtracing

} // visionaray

#include "detail/pathtracing.inl"
#include "detail/simple.inl"
#include "detail/whitted.inl"

#endif // VSNRAY_KERNELS_H


