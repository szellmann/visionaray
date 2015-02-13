// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_KERNELS_H
#define VSNRAY_KERNELS_H

#include <visionaray/scheduler.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Param structs
//

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

} // visionaray

#include "detail/pathtracing.inl"
#include "detail/simple.inl"
#include "detail/whitted.inl"

#endif // VSNRAY_KERNELS_H


