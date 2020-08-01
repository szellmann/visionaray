// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_KERNELS_H
#define VSNRAY_KERNELS_H 1

#include <cstddef>
#include <iterator>
#include <limits>

#include "math/forward.h"
#include "math/vector.h"
#include "prim_traits.h"
#include "tags.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Parameter struct for built-in kernels
//

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
    typename EnvMap
    >
struct kernel_params
{
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

    Normals   geometric_normals;
    Normals   shading_normals;
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

    EnvMap environment_map;
};


//-------------------------------------------------------------------------------------------------
// Factory for param struct
//

// default ------------------------------------------------

template <
    typename Primitives,
    typename Materials
    >
auto make_kernel_params(
        Primitives const&   begin,
        Primitives const&   end,
        Materials const&    materials,
        unsigned            num_bounces     = 5,
        float               epsilon         = std::numeric_limits<float>::epsilon(),
        vec4 const&         bg_color        = vec4(0.0),
        vec4 const&         ambient_color   = vec4(0.0)
        )
    -> kernel_params<
        unspecified_binding,
        unspecified_binding,
        Primitives,
        vector<3, typename scalar_type<typename std::iterator_traits<Primitives>::value_type>::type>*,
        vector<2, typename scalar_type<typename std::iterator_traits<Primitives>::value_type>::type>*,
        Materials,
        vector<3, typename scalar_type<typename std::iterator_traits<Primitives>::value_type>::type>*,
        std::nullptr_t*, // dummy texture type
        std::nullptr_t*, // dummy light type
        vec4,
        std::nullptr_t*  // dummy env map type
        >
{
    return {
        { begin, end },
        nullptr, // geometric normals
        nullptr, // shading normals
        nullptr, // uvs
        materials,
        nullptr, // colors
        nullptr, // textures
        { nullptr, nullptr }, // lights
        num_bounces,
        epsilon,
        bg_color,
        ambient_color,
        nullptr  // env map
        };
}


// w/ lights ----------------------------------------------

template <
    typename Primitives,
    typename Materials,
    typename Lights
    >
auto make_kernel_params(
        Primitives const&   begin,
        Primitives const&   end,
        Materials const&    materials,
        Lights const&       lbegin,
        Lights const&       lend,
        unsigned            num_bounces     = 5,
        float               epsilon         = std::numeric_limits<float>::epsilon(),
        vec4 const&         bg_color        = vec4(0.0),
        vec4 const&         ambient_color   = vec4(0.0)
        )
    -> kernel_params<
        unspecified_binding,
        unspecified_binding,
        Primitives,
        vector<3, typename scalar_type<typename std::iterator_traits<Primitives>::value_type>::type>*,
        vector<2, typename scalar_type<typename std::iterator_traits<Primitives>::value_type>::type>*,
        Materials,
        vector<3, typename scalar_type<typename std::iterator_traits<Primitives>::value_type>::type>*,
        std::nullptr_t*, // dummy texture type
        Lights,
        vec4,
        std::nullptr_t*  // dummy env map type
        >
{
    return {
        { begin, end },
        nullptr, // geometric normals
        nullptr, // shading normals
        nullptr, // uvs
        materials,
        nullptr, // colors
        nullptr, // textures
        { lbegin, lend },
        num_bounces,
        epsilon,
        bg_color,
        ambient_color,
        nullptr  // env map
        };
}


// w/ normals ---------------------------------------------

template <
    typename NormalBinding,
    typename Primitives,
    typename Normals,
    typename Materials,
    typename Lights,
    typename = typename std::enable_if<is_normal_binding<NormalBinding>::value>::type
    >
auto make_kernel_params(
        NormalBinding       /* */,
        Primitives const&   begin,
        Primitives const&   end,
        Normals const&      geometric_normals,
        Normals const&      shading_normals,
        Materials const&    materials,
        Lights const&       lbegin,
        Lights const&       lend,
        unsigned            num_bounces     = 5,
        float               epsilon         = std::numeric_limits<float>::epsilon(),
        vec4 const&         bg_color        = vec4(0.0),
        vec4 const&         ambient_color   = vec4(0.0)
        )
    -> kernel_params<
        NormalBinding,
        unspecified_binding,
        Primitives,
        Normals,
        vector<2, typename scalar_type<typename std::iterator_traits<Primitives>::value_type>::type>*,
        Materials,
        vector<3, typename scalar_type<typename std::iterator_traits<Primitives>::value_type>::type>*,
        std::nullptr_t*, // dummy texture type
        Lights,
        vec4,
        std::nullptr_t*  // dummy env map type
        >
{
    return {
        { begin, end },
        geometric_normals,
        shading_normals,
        nullptr, // uvs
        materials,
        nullptr, // colors
        nullptr, // textures
        { lbegin, lend },
        num_bounces,
        epsilon,
        bg_color,
        ambient_color,
        nullptr  // env map
        };
}


// w/ normals and textures --------------------------------

template <
    typename NormalBinding,
    typename Primitives,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Textures,
    typename Lights,
    typename = typename std::enable_if<is_normal_binding<NormalBinding>::value>::type
    >
auto make_kernel_params(
        NormalBinding       /* */,
        Primitives const&   begin,
        Primitives const&   end,
        Normals const&      geometric_normals,
        Normals const&      shading_normals,
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
    -> kernel_params<
        NormalBinding,
        unspecified_binding,
        Primitives,
        Normals,
        TexCoords,
        Materials,
        vector<3, typename scalar_type<typename std::iterator_traits<Primitives>::value_type>::type>*,
        Textures,
        Lights,
        vec4,
        std::nullptr_t* // dummy env map type
        >
{
    return {
        { begin, end },
        geometric_normals,
        shading_normals,
        tex_coords,
        materials,
        nullptr, // colors
        textures,
        { lbegin, lend },
        num_bounces,
        epsilon,
        bg_color,
        ambient_color,
        nullptr  // env map
        };
}


// w/ normals, colors and textures ------------------------

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
    typename = typename std::enable_if<is_normal_binding<NormalBinding>::value>::type,
    typename = typename std::enable_if<is_color_binding<ColorBinding>::value>::type
    >
auto make_kernel_params(
        NormalBinding       /* */,
        ColorBinding        /* */,
        Primitives const&   begin,
        Primitives const&   end,
        Normals const&      geometric_normals,
        Normals const&      shading_normals,
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
        vec4,
        std::nullptr_t* // dummy env map type
        >
{
    return {
        { begin, end },
        geometric_normals,
        shading_normals,
        tex_coords,
        materials,
        colors,
        textures,
        { lbegin, lend },
        num_bounces,
        epsilon,
        bg_color,
        ambient_color,
        nullptr // env map
        };
}

// w/ everything ------------------------------------------

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
    typename EnvMap,
    typename = typename std::enable_if<is_normal_binding<NormalBinding>::value>::type,
    typename = typename std::enable_if<is_color_binding<ColorBinding>::value>::type
    >
auto make_kernel_params(
        NormalBinding       /* */,
        ColorBinding        /* */,
        Primitives const&   begin,
        Primitives const&   end,
        Normals const&      geometric_normals,
        Normals const&      shading_normals,
        TexCoords const&    tex_coords,
        Materials const&    materials,
        Colors const&       colors,
        Textures const&     textures,
        Lights const&       lbegin,
        Lights const&       lend,
        EnvMap const&       environment_map,
        unsigned            num_bounces     = 5,
        float               epsilon         = std::numeric_limits<float>::epsilon()
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
        vec4,
        EnvMap
        >
{
    return {
        { begin, end },
        geometric_normals,
        shading_normals,
        tex_coords,
        materials,
        colors,
        textures,
        { lbegin, lend },
        num_bounces,
        epsilon,
        vec4(), // dummy bgcolor
        vec4(), // ambient color
        environment_map
        };
}
} // visionaray

#include "detail/pathtracing.inl"
#include "detail/simple.inl"
#include "detail/whitted.inl"

#endif // VSNRAY_KERNELS_H
