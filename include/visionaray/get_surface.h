// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GET_SURFACE_H
#define VSNRAY_GET_SURFACE_H 1

#include <type_traits>
#include <utility>

#include "texture/texture_traits.h"
#include "array.h"
#include "bvh.h"
#include "get_color.h"
#include "get_normal.h"
#include "get_primitive.h"
#include "get_shading_normal.h"
#include "get_tex_coord.h"
#include "surface.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Helper functions
//

// deduce simd surface type from params -------------------

template <typename Params, typename T>
class simd_decl_surface
{
    enum { Size_ = simd::num_elements<T>::value };
    using N_     = typename Params::normal_type;
    using C_     = typename Params::color_type;
    using M_     = typename Params::material_type;

public:
    using type = surface<
        decltype(simd::pack(std::declval<array<N_, Size_>>())),
        decltype(simd::pack(std::declval<array<C_, Size_>>())),
        decltype(simd::pack(std::declval<array<M_, Size_>>()))
        >;

    using array_type = array<surface<N_, C_, M_>, Size_>;
};


//-------------------------------------------------------------------------------------------------
// Sample textures
//

template <typename HR, typename Params>
VSNRAY_FUNC
inline typename Params::color_type get_tex_color(
        HR const&                      hr,
        Params const&                  params,
        std::integral_constant<int, 0> /* not a texture! */
        )
{
    VSNRAY_UNUSED(hr);
    VSNRAY_UNUSED(params);

    using C = typename Params::color_type;

    // Just return white
    return C(1.0);
}

template <typename HR, typename Params>
VSNRAY_FUNC
inline typename Params::color_type get_tex_color(
        HR const&                      hr,
        Params const&                  params,
        std::integral_constant<int, 1> /* */
        )
{
    using C = typename Params::color_type;

    auto coord = get_tex_coord(params.tex_coords, hr, get_primitive(params, hr));

    int mat_id = hr.inst_id < 0 ? hr.geom_id : hr.inst_id;
    auto const& tex = params.textures[mat_id];
    return C(tex1D(tex, coord));
}

template <typename HR, typename Params>
VSNRAY_FUNC
inline typename Params::color_type get_tex_color(
        HR const&                      hr,
        Params const&                  params,
        std::integral_constant<int, 2> /* */
        )
{
    using C = typename Params::color_type;

    auto coord = get_tex_coord(params.tex_coords, hr, get_primitive(params, hr));

    int mat_id = hr.inst_id < 0 ? hr.geom_id : hr.inst_id;
    auto const& tex = params.textures[mat_id];
    return C(tex2D(tex, coord));
}

template <typename HR, typename Params>
VSNRAY_FUNC
inline typename Params::color_type get_tex_color(
        HR const&                      hr,
        Params const&                  params,
        std::integral_constant<int, 3> /* */
        )
{
    using C = typename Params::color_type;

    auto coord = get_tex_coord(params.tex_coords, hr, get_primitive(params, hr));

    int mat_id = hr.inst_id < 0 ? hr.geom_id : hr.inst_id;
    auto const& tex = params.textures[mat_id];
    return C(tex3D(tex, coord));
}


//-------------------------------------------------------------------------------------------------
// No SIMD
//

template <
    typename HR,
    typename Params,
    typename = typename std::enable_if<!simd::is_simd_vector<typename HR::scalar_type>::value>::type
    >
VSNRAY_FUNC
inline auto get_surface_impl(HR const& hr, Params const& params)
    -> surface<
            typename Params::normal_type,
            typename Params::color_type,
            typename Params::material_type
            >
{
    using C = typename Params::color_type;

    auto const& prim = get_primitive(params, hr);

    auto const& gns = params.geometric_normals;
    auto const& sns = params.shading_normals;

    auto gn    = gns ? get_normal(gns, hr, prim) : get_normal(hr, prim);
    auto sn    = sns ? get_shading_normal(sns, hr, prim, typename Params::normal_binding{}) : gn;
    auto color = params.colors ? get_color(params.colors, hr, prim, typename Params::color_binding{}) : C(1.0);
    auto tc    = params.tex_coords && params.textures ? get_tex_color(
                        hr,
                        params,
                        std::integral_constant<int, texture_dimensions<typename Params::texture_type>::value>{}
                        ) : C(1.0);

    int mat_id = hr.inst_id < 0 ? hr.geom_id : hr.inst_id;
    return { gn, sn, color * tc, params.materials[mat_id] };
}


//-------------------------------------------------------------------------------------------------
// SIMD
//

template <
    typename HR,
    typename Params,
    typename = typename std::enable_if<simd::is_simd_vector<typename HR::scalar_type>::value>::type
    >
VSNRAY_FUNC
inline auto get_surface_impl(HR const& hr, Params const& params)
{
    using T = typename HR::scalar_type;

    auto hrs = unpack(hr);

    typename simd_decl_surface<Params, T>::array_type surfs;

    for (int i = 0; i < simd::num_elements<T>::value; ++i)
    {
        if (hrs[i].hit)
        {
            surfs[i] = get_surface_impl(hrs[i], params);
        }
    }

    return simd::pack(surfs);
}

} // detail


template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface(HR const& hr, Params const& p)
{
    return detail::get_surface_impl(hr, p);
}

} // visionaray

#endif // VSNRAY_SURFACE_H
