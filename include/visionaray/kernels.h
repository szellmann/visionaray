// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_KERNELS_H
#define VSNRAY_KERNELS_H

#include <visionaray/math/math.h>
#include <visionaray/scheduler.h>
#include <visionaray/tags.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Param structs
//

template <typename ...Args>
struct kernel_params;

template <typename NB, typename P, typename N, typename M, typename L, typename C, typename ...Args>
struct kernel_params<NB, P, N, M, L, C, Args...>
{
    using normal_binding = NB;

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

    C bg_color;
    C ambient_color;
};

template <typename NB, typename P, typename N, typename TC, typename M, typename T, typename L, typename C, typename ...Args>
struct kernel_params<NB, P, N, TC, M, T, L, C, Args...>
{
    using normal_binding = NB;

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

    C bg_color;
    C ambient_color;
};


//-------------------------------------------------------------------------------------------------
// Factory for param structs
//

template <typename NB, typename P, typename N, typename M, typename L>
kernel_params<NB, P, N, M, L, vec4>  make_params(P const& begin, P const& end, N const& normals,
    M const& materials, L const& lbegin, L const& lend,
    vec4 const& bg_color = vec4(0.0, 0.0, 0.0, 1.0),
    vec4 const& ambient_color = vec4(1.0, 1.0, 1.0, 1.0))
{
    return kernel_params<NB, P, N, M, L, vec4>
    {
        { begin, end }, normals, materials, { lbegin, lend },
        bg_color, ambient_color
    };
}

template <typename NB, typename P, typename N, typename TC, typename M, typename T, typename L>
kernel_params<NB, P, N, TC, M, T, L, vec4> make_params(P const& begin, P const& end, N const& normals,
    TC const& tex_coords, M const& materials, T const& textures, L const& lbegin, L const& lend,
    vec4 const& bg_color = vec4(0.0, 0.0, 0.0, 1.0),
    vec4 const& ambient_color = vec4(1.0, 1.0, 1.0, 1.0))
{
    return kernel_params<NB, P, N, TC, M, T, L, vec4>
    {
        { begin, end }, normals, tex_coords, materials, textures, { lbegin, lend },
        bg_color, ambient_color
    };
}

} // visionaray

#include "detail/pathtracing.inl"
#include "detail/simple.inl"
#include "detail/whitted.inl"

#endif // VSNRAY_KERNELS_H
