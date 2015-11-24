// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_KERNELS_H
#define VSNRAY_KERNELS_H 1

#include <iterator>
#include <limits>

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

template <
    typename NormalBinding,
    typename Primitives,
    typename Normals,
    typename Materials,
    typename Lights,
    typename Color,
    typename ...Args
    >
struct kernel_params<
        NormalBinding,
        Primitives,
        Normals,
        Materials,
        Lights,
        Color,
        Args...
        >
{
    using normal_binding    = NormalBinding;
    using primitive_type    = typename std::iterator_traits<Primitives>::value_type;
    using normal_type       = typename std::iterator_traits<Normals>::value_type;
    using material_type     = typename std::iterator_traits<Materials>::value_type;
    using light_type        = typename std::iterator_traits<Lights>::value_type;

    struct
    {
        Primitives begin;
        Primitives end;
    } prims;

    Normals   normals;
    Materials materials;

    struct
    {
        Lights begin;
        Lights end;
    } lights;

    unsigned num_bounces;
    float epsilon;

    Color bg_color;
    Color ambient_color;
};

template <
    typename NormalBinding,
    typename Primitives,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Textures,
    typename Lights,
    typename Color,
    typename ...Args
    >
struct kernel_params<
        NormalBinding,
        Primitives,
        Normals,
        TexCoords,
        Materials,
        Textures,
        Lights,
        Color,
        Args...
        >
{
    using has_textures      = void;

    using normal_binding    = NormalBinding;
    using primitive_type    = typename std::iterator_traits<Primitives>::value_type;
    using normal_type       = typename std::iterator_traits<Normals>::value_type;
    using tex_coords_type   = typename std::iterator_traits<TexCoords>::value_type;
    using material_type     = typename std::iterator_traits<Materials>::value_type;
    using texture_type      = typename std::iterator_traits<Textures>::value_type;
    using light_type        = typename std::iterator_traits<Lights>::value_type;

    struct
    {
        Primitives begin;
        Primitives end;
    } prims;

    Normals   normals;
    TexCoords tex_coords;
    Materials materials;
    Textures  textures;

    struct
    {
        Lights begin;
        Lights end;
    } lights;

    unsigned num_bounces;
    float epsilon;

    Color bg_color;
    Color ambient_color;
};

template <
    typename NormalBinding,
    typename ColorBinding,
    typename Primitives,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Colors,
    typename Textures,
    typename Lights,
    typename Color,
    typename ...Args
    >
struct kernel_params<
        NormalBinding,
        ColorBinding,
        Primitives,
        Normals,
        TexCoords,
        Materials,
        Colors,
        Textures,
        Lights,
        Color,
        Args...
        >
{
    using has_textures      = void;
    using has_colors        = void;

    using normal_binding    = NormalBinding;
    using color_binding     = ColorBinding;
    using primitive_type    = typename std::iterator_traits<Primitives>::value_type;
    using normal_type       = typename std::iterator_traits<Normals>::value_type;
    using tex_coords_type   = typename std::iterator_traits<TexCoords>::value_type;
    using material_type     = typename std::iterator_traits<Materials>::value_type;
    using color_type        = typename std::iterator_traits<Colors>::value_type;
    using texture_type      = typename std::iterator_traits<Textures>::value_type;
    using light_type        = typename std::iterator_traits<Lights>::value_type;

    struct
    {
        Primitives begin;
        Primitives end;
    } prims;

    Normals   normals;
    TexCoords tex_coords;
    Materials materials;
    Colors    colors;
    Textures  textures;

    struct
    {
        Lights begin;
        Lights end;
    } lights;

    unsigned num_bounces;
    float epsilon;

    Color bg_color;
    Color ambient_color;
};


//-------------------------------------------------------------------------------------------------
// Factory for param structs
//

template <
    typename NormalBinding,
    typename Primitives,
    typename Normals,
    typename Materials,
    typename Lights
    >
auto make_kernel_params(
        NormalBinding       /* */,
        Primitives const&   begin,
        Primitives const&   end,
        Normals const&      normals,
        Materials const&    materials,
        Lights const&       lbegin,
        Lights const&       lend,
        unsigned            num_bounces     = 5,
        float               epsilon         = std::numeric_limits<float>::epsilon(),
        vec4 const&         bg_color        = vec4(0.0),
        vec4 const&         ambient_color   = vec4(0.0)
        )
    -> kernel_params<NormalBinding, Primitives, Normals, Materials, Lights, vec4>
{
    return kernel_params<NormalBinding, Primitives, Normals, Materials, Lights, vec4>{
            { begin, end },
            normals,
            materials,
            { lbegin, lend },
            num_bounces,
            epsilon,
            bg_color,
            ambient_color
            };
}

template <
    typename NormalBinding,
    typename Primitives,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Textures,
    typename Lights
    >
auto make_kernel_params(
        NormalBinding       /* */,
        Primitives const&   begin,
        Primitives const&   end,
        Normals const&      normals,
        TexCoords const&    tex_coords,
        Materials const&    materials,
        Textures const&     textures,
        Lights const&       lbegin,
        Lights const&       lend,
        unsigned            num_bounces     = 5,
        float               epsilon         = std::numeric_limits<float>::epsilon(),
        vec4 const&         bg_color        = vec4(0.0),
        vec4 const&         ambient_color   = vec4(1.0)
        )
    -> kernel_params<NormalBinding, Primitives, Normals, TexCoords, Materials, Textures, Lights, vec4>
{
    return kernel_params<NormalBinding, Primitives, Normals, TexCoords, Materials, Textures, Lights, vec4>{
            { begin, end },
            normals,
            tex_coords,
            materials,
            textures,
            { lbegin, lend },
            num_bounces,
            epsilon,
            bg_color,
            ambient_color
            };
}

template <
    typename NormalBinding,
    typename ColorBinding,
    typename Primitives,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Colors,
    typename Textures,
    typename Lights
    >
auto make_kernel_params(
        NormalBinding       /* */,
        ColorBinding        /* */,
        Primitives const&   begin,
        Primitives const&   end,
        Normals const&      normals,
        TexCoords const&    tex_coords,
        Materials const&    materials,
        Colors const&       colors,
        Textures const&     textures,
        Lights const&       lbegin,
        Lights const&       lend,
        unsigned            num_bounces     = 5,
        float               epsilon         = std::numeric_limits<float>::epsilon(),
        vec4 const&         bg_color        = vec4(0.0),
        vec4 const&         ambient_color   = vec4(1.0)
        )
    -> kernel_params<
        NormalBinding,
        ColorBinding,
        Primitives,
        Normals,
        TexCoords,
        Materials,
        Colors,
        Textures,
        Lights,
        vec4
        >
{
    return kernel_params<
        NormalBinding,
        ColorBinding,
        Primitives,
        Normals,
        TexCoords,
        Materials,
        Colors,
        Textures,
        Lights,
        vec4
        >{
            { begin, end },
            normals,
            tex_coords,
            materials,
            colors,
            textures,
            { lbegin, lend },
            num_bounces,
            epsilon,
            bg_color,
            ambient_color
            };
}

} // visionaray

#include "detail/pathtracing.inl"
#include "detail/simple.inl"
#include "detail/whitted.inl"

#endif // VSNRAY_KERNELS_H
