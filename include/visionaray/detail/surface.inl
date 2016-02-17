// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SURFACE_INL
#define VSNRAY_SURFACE_INL 1

#include <array>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "../generic_material.h"
#include "../generic_primitive.h"
#include "../get_color.h"
#include "../get_normal.h"
#include "../get_tex_coord.h"
#include "../prim_traits.h"
#include "../tags.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Helper functions
//

template <typename N, typename M>
VSNRAY_FUNC
surface<N, M> make_surface(N const& n, M const& m)
{
    return surface<N, M>(n, m);
}

template <typename N, typename M, typename C>
VSNRAY_FUNC
surface<N, M, C> make_surface(N const& n, M const m, C const& tex_color)
{
    return surface<N, M, C>(n, m, tex_color);
}


// deduce surface type from params ------------------------

template <typename ...Args>
struct decl_surface;

template <
    typename Normals,
    typename Materials
    >
struct decl_surface<Normals, Materials>
{
    using type = surface<
        typename std::iterator_traits<Normals>::value_type,
        typename std::iterator_traits<Materials>::value_type
        >;
};

template <
    typename Normals,
    typename Materials,
    typename DiffuseColor
    >
struct decl_surface<Normals, Materials, DiffuseColor>
{
    using type = surface<
        typename std::iterator_traits<Normals>::value_type,
        typename std::iterator_traits<Materials>::value_type,
        DiffuseColor
        >;
};


// identify accelerators ----------------------------------

template <typename T>
struct primitive_traits
{
    using type = T;
};

template <template <typename> class Accelerator, typename T>
struct primitive_traits<Accelerator<T>>
{
    using type = T;
};


// TODO: consolidate the following with get_normal()?
// TODO: consolidate interface with get_color() and get_tex_coord()?

// dispatch function for get_normal() ---------------------

// overload w/ precalculated normals
template <
    typename Primitives,
    typename Normals,
    typename HR,
    typename Primitive,
    typename NormalBinding,
    typename = typename std::enable_if<num_normals<Primitive, NormalBinding>::value>::type
    >
VSNRAY_FUNC
inline auto get_normal_dispatch(
        Primitives      primitives,
        Normals         normals,
        HR const&       hr,
        Primitive       /* */,
        NormalBinding   /* */
        )
    -> decltype( get_normal(normals, hr, Primitive{}, NormalBinding{}) )
{
    VSNRAY_UNUSED(primitives);

    return get_normal(normals, hr, Primitive{}, NormalBinding{});
}

// overload w/o precalculated normals
template <
    typename Primitives,
    typename Normals,
    typename HR,
    typename Primitive,
    typename NormalBinding,
    typename = typename std::enable_if<num_normals<Primitive, NormalBinding>::value == 0>::type
    >
VSNRAY_FUNC
inline auto get_normal_dispatch(
        Primitives      primitives,
        Normals         normals,
        HR const&       hr,
        Primitive       /* */,
        NormalBinding   /* */
        )
    -> decltype( get_normal(hr, primitives[hr.prim_id], NormalBinding{}) )
{
    VSNRAY_UNUSED(normals);

    return get_normal(hr, primitives[hr.prim_id], NormalBinding{});
}

// helper
template <typename NormalBinding, typename Normals, typename HR>
class get_normal_from_generic_primitive_visitor
{
public:

    using return_type = vector<3, float>;

public:

    VSNRAY_FUNC
    get_normal_from_generic_primitive_visitor(
            Normals     normals,
            HR const&   hr
            )
        : normals_(normals)
        , hr_(hr)
    {
    }

    // TODO: with get_normal() having a streamlined interface,
    // we won't require explicit specializations for planes and spheres,
    // thus making 'generic_primitive' really generic..
    VSNRAY_FUNC
    return_type operator()(basic_plane<3, float> const& plane) const
    {
        return get_normal(hr_, plane, NormalBinding{}); // TODO
    }

    VSNRAY_FUNC
    return_type operator()(basic_sphere<float> const& sphere) const
    {
        return get_normal(hr_, sphere, NormalBinding{}); // TODO
    }

    template <typename Primitive>
    VSNRAY_FUNC
    return_type operator()(Primitive const& primitive) const
    {
        VSNRAY_UNUSED(primitive);

        return get_normal(normals_, hr_, primitive, NormalBinding{});
    }

private:

    Normals     normals_;
    HR const&   hr_;

};

// overload for generic_primitive
template <
    typename Primitives,
    typename Normals,
    typename HR,
    typename ...Ts,
    typename NormalBinding
    >
VSNRAY_FUNC
inline auto get_normal_dispatch(
        Primitives                  primitives,
        Normals                     normals,
        HR const&                   hr,
        generic_primitive<Ts...>    /* */,
        NormalBinding               /* */
        )
    -> typename std::iterator_traits<Normals>::value_type
{
    get_normal_from_generic_primitive_visitor<NormalBinding, Normals, HR> visitor(
            normals,
            hr
            );

    return apply_visitor( visitor, primitives[hr.prim_id] );
}


} // detail


namespace simd
{

//-------------------------------------------------------------------------------------------------
// Functions to pack and unpack SIMD surfaces
//

template <typename N, typename M, typename ...Args, size_t Size>
inline auto pack(std::array<surface<N, M, Args...>, Size> const& surfs)
    -> decltype( visionaray::detail::make_surface(
            pack(std::declval<std::array<N, Size>>()),
            pack(std::declval<std::array<M, Size>>())
            ) )
{
    std::array<N, Size> normals;
    std::array<M, Size> materials;

    for (size_t i = 0; i < Size; ++i)
    {
        normals[i]      = surfs[i].normal;
        materials[i]    = surfs[i].material;
    }

    return visionaray::detail::make_surface(
            pack(normals),
            pack(materials)
            );
}

template <typename N, typename M, typename C, typename ...Args, size_t Size>
inline auto pack(std::array<surface<N, M, C, Args...>, Size> const& surfs)
    -> decltype( visionaray::detail::make_surface(
            pack(std::declval<std::array<N, Size>>()),
            pack(std::declval<std::array<M, Size>>()),
            pack(std::declval<std::array<C, Size>>())
            ) )
{
    std::array<N, Size> normals;
    std::array<M, Size> materials;
    std::array<C, Size> tex_colors;

    for (size_t i = 0; i < Size; ++i)
    {
        normals[i]      = surfs[i].normal;
        materials[i]    = surfs[i].material;
        tex_colors[i]   = surfs[i].tex_color_;
    }

    return visionaray::detail::make_surface(
            pack(normals),
            pack(materials),
            pack(tex_colors)
            );
}

} // simd


//-------------------------------------------------------------------------------------------------
//
//

template <
    typename HR,
    typename Primitives,
    typename Normals,
    typename Materials,
    typename Primitive,
    typename NormalBinding
    >
VSNRAY_FUNC
inline auto get_surface_any_prim_impl(
        HR const&       hr,
        Primitives      primitives,
        Normals         normals,
        Materials       materials,
        Primitive       /* */,
        NormalBinding   /* */
        )
    -> typename detail::decl_surface<Normals, Materials>::type
{
    return detail::make_surface(
            detail::get_normal_dispatch(primitives, normals, hr, Primitive{}, NormalBinding{}),
            materials[hr.geom_id]
            );
}

template <
    typename HR,
    typename Primitives,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Textures,
    typename Primitive,
    typename NormalBinding
    >
VSNRAY_FUNC
inline auto get_surface_any_prim_impl(
        HR const&       hr,
        Primitives      primitives,
        Normals         normals,
        TexCoords       tex_coords,
        Materials       materials,
        Textures        textures,
        Primitive       /* */,
        NormalBinding   /* */
        )
    -> typename detail::decl_surface<Normals, Materials, vector<3, float>>::type
{
    using P = typename detail::primitive_traits<Primitive>::type;
    using C = vector<3, float>;

    auto tc = get_tex_coord(tex_coords, hr, P{});

    auto const& tex = textures[hr.geom_id];
    auto tex_color = tex.width() > 0 && tex.height() > 0
                   ? C(tex2D(tex, tc))
                   : C(1.0);

    auto normal = detail::get_normal_dispatch(primitives, normals, hr, P{}, NormalBinding{});
    return detail::make_surface( normal, materials[hr.geom_id], tex_color );
}

template <
    typename HR,
    typename Primitives,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Colors,
    typename Textures,
    typename Primitive,
    typename NormalBinding,
    typename ColorBinding
    >
VSNRAY_FUNC
inline auto get_surface_any_prim_impl(
        HR const&       hr,
        Primitives      primitives,
        Normals         normals,
        TexCoords       tex_coords,
        Materials       materials,
        Colors          colors,
        Textures        textures,
        Primitive       /* */,
        NormalBinding   /* */,
        ColorBinding    /* */
        )
    -> typename detail::decl_surface<Normals, Materials, vector<3, float>>::type
{
    using P = typename detail::primitive_traits<Primitive>::type;
    using C = vector<3, float>;

    auto color = get_color(colors, hr, P{}, ColorBinding{});
    auto tc = get_tex_coord(tex_coords, hr, P{});

    auto const& tex = textures[hr.geom_id];
    auto tex_color = tex.width() > 0 && tex.height() > 0
                   ? C(tex2D(tex, tc))
                   : C(1.0);

    auto normal = detail::get_normal_dispatch(primitives, normals, hr, P{}, NormalBinding{});
    return detail::make_surface( normal, materials[hr.geom_id], color * tex_color );
}


//-------------------------------------------------------------------------------------------------
// Generic primitive / float
//

namespace detail
{

} // detail


//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / simd type
//

template <
    template <typename, typename> class HR,
    typename T,
    typename HRP,
    typename Primitives,
    typename Normals,
    typename Materials,
    typename Primitive,
    typename NormalBinding,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
inline auto get_surface_any_prim_impl(
        HR<basic_ray<T>, HRP> const&    hr,
        Primitives                      primitives,
        Normals                         normals,
        Materials                       materials,
        Primitive                       /* */,
        NormalBinding                   /* */
        ) -> decltype( simd::pack(
            std::declval<std::array<
                typename detail::decl_surface<Normals, Materials>::type,
                simd::num_elements<T>::value
                >>()
            ) )
{
    using N = typename std::iterator_traits<Normals>::value_type;
    using M = typename std::iterator_traits<Materials>::value_type;
    using P = typename detail::primitive_traits<Primitive>::type;

    auto hrs = unpack(hr);

    std::array<typename detail::decl_surface<Normals, Materials>::type, simd::num_elements<T>::value> surfs;

    for (int i = 0; i < simd::num_elements<T>::value; ++i)
    {
        surfs[i] = detail::make_surface(
            hrs[i].hit ? detail::get_normal_dispatch(primitives, normals, hrs[i], P{}, NormalBinding{}) : N{},
            hrs[i].hit ? materials[hrs[i].geom_id]                                                      : M{}
            );
    }

    return simd::pack(surfs);
}


template <
    template <typename, typename> class HR,
    typename T,
    typename HRP,
    typename Primitives,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Textures,
    typename Primitive,
    typename NormalBinding,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
inline auto get_surface_any_prim_impl(
        HR<basic_ray<T>, HRP> const&    hr,
        Primitives                      primitives,
        Normals                         normals,
        TexCoords                       tex_coords,
        Materials                       materials,
        Textures                        textures,
        Primitive                       /* */,
        NormalBinding                   /* */
        ) -> decltype( simd::pack(
            std::declval<std::array<
                typename detail::decl_surface<Normals, Materials, vector<3, float>>::type,
                simd::num_elements<T>::value
                >>()
            ) )
{
    using N = typename std::iterator_traits<Normals>::value_type;
    using M = typename std::iterator_traits<Materials>::value_type;
    using P = typename detail::primitive_traits<Primitive>::type;
    using C = vector<3, float>;

    auto hrs = unpack(hr);

    auto tcs = get_tex_coord(tex_coords, hrs, typename detail::primitive_traits<Primitive>::type{});

    std::array<typename detail::decl_surface<Normals, Materials, vector<3, float>>::type, simd::num_elements<T>::value> surfs;

    for (int i = 0; i < simd::num_elements<T>::value; ++i)
    {
        auto const& tex = textures[hrs[i].geom_id];
        C tex_color = hrs[i].hit && tex.width() > 0 && tex.height() > 0
                    ? C(tex2D(tex, tcs[i]))
                    : C(1.0);

        surfs[i] = detail::make_surface(
            hrs[i].hit ? detail::get_normal_dispatch(primitives, normals, hrs[i], P{}, NormalBinding{}) : N{},
            hrs[i].hit ? materials[hrs[i].geom_id]                                                      : M{},
            hrs[i].hit ? tex_color                                                                      : C(1.0)
            );
    }

    return simd::pack(surfs);
}

template <
    template <typename, typename> class HR,
    typename T,
    typename HRP,
    typename Primitives,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Colors,
    typename Textures,
    typename Primitive,
    typename NormalBinding,
    typename ColorBinding,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
inline auto get_surface_any_prim_impl(
        HR<basic_ray<T>, HRP> const&    hr,
        Primitives                      primitives,
        Normals                         normals,
        TexCoords                       tex_coords,
        Materials                       materials,
        Colors                          colors,
        Textures                        textures,
        Primitive                       /* */,
        NormalBinding                   /* */,
        ColorBinding                    /* */
        ) -> decltype( simd::pack(
            std::declval<std::array<
                typename detail::decl_surface<Normals, Materials, vector<3, float>>::type,
                simd::num_elements<T>::value
                >>()
            ) )
{
    using N = typename std::iterator_traits<Normals>::value_type;
    using M = typename std::iterator_traits<Materials>::value_type;
    using P = typename detail::primitive_traits<Primitive>::type;
    using C = vector<3, float>;

    auto hrs = unpack(hr);

    auto colorss = get_color(colors, hrs, typename detail::primitive_traits<Primitive>::type{}, ColorBinding{});

    auto tcs = get_tex_coord(tex_coords, hrs, typename detail::primitive_traits<Primitive>::type{});

    std::array<typename detail::decl_surface<Normals, Materials, vector<3, float>>::type, simd::num_elements<T>::value> surfs;

    for (int i = 0; i < simd::num_elements<T>::value; ++i)
    {
        auto const& tex = textures[hrs[i].geom_id];
        C tex_color = hrs[i].hit && tex.width() > 0 && tex.height() > 0
                    ? C(tex2D(tex, tcs[i]))
                    : C(1.0);

        surfs[i] = detail::make_surface(
            hrs[i].hit ? detail::get_normal_dispatch(primitives, normals, hrs[i], P{}, NormalBinding{}) : N{},
            hrs[i].hit ? materials[hrs[i].geom_id]                                                      : M{},
            hrs[i].hit ? colorss[i] * tex_color                                                         : C(1.0)
            );
    }

    return simd::pack(surfs);
}


//-------------------------------------------------------------------------------------------------
// Functions to deduce appropriate surface via parameter inspection
//

template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface_unroll_params_impl(
        HR const& hr,
        Params const& p,
        detail::has_no_colors_tag,
        detail::has_no_textures_tag
        )
    -> decltype( get_surface_any_prim_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.materials,
            typename Params::primitive_type{},
            typename Params::normal_binding{}
            ) )
{
    return get_surface_any_prim_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.materials,
            typename Params::primitive_type{},
            typename Params::normal_binding{}
            );
}

template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface_unroll_params_impl(
        HR const& hr,
        Params const& p,
        detail::has_no_colors_tag,
        detail::has_textures_tag
        )
    -> decltype( get_surface_any_prim_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.tex_coords,
            p.materials,
            p.textures,
            typename Params::primitive_type{},
            typename Params::normal_binding{}
            ) )
{
    return get_surface_any_prim_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.tex_coords,
            p.materials,
            p.textures,
            typename Params::primitive_type{},
            typename Params::normal_binding{}
            );
}

template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface_unroll_params_impl(
        HR const& hr,
        Params const& p,
        detail::has_colors_tag,
        detail::has_textures_tag
        )
    -> decltype( get_surface_any_prim_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.tex_coords,
            p.materials,
            p.colors,
            p.textures,
            typename Params::primitive_type{},
            typename Params::normal_binding{},
            typename Params::color_binding{}
            ) )
{
    return get_surface_any_prim_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.tex_coords,
            p.materials,
            p.colors,
            p.textures,
            typename Params::primitive_type{},
            typename Params::normal_binding{},
            typename Params::color_binding{}
            );
}


template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface(HR const& hr, Params const& p)
    -> decltype( get_surface_unroll_params_impl(
            hr,
            p,
            detail::has_colors<Params>{},
            detail::has_textures<Params>{}
            ) )
{
    return get_surface_unroll_params_impl(
            hr,
            p,
            detail::has_colors<Params>{},
            detail::has_textures<Params>{}
            );
}


//-------------------------------------------------------------------------------------------------
// General surface functions
//

template <typename ...Args>
VSNRAY_FUNC
inline bool has_emissive_material(surface<Args...> const& surf)
{
    VSNRAY_UNUSED(surf);
    return false;
}

template <typename N, typename T>
VSNRAY_FUNC
inline bool has_emissive_material(surface<N, emissive<T>> const& surf)
{
    VSNRAY_UNUSED(surf);
    return true;
}

template <typename N, typename ...Ms, typename ...Ts>
VSNRAY_FUNC
inline auto has_emissive_material(surface<N, generic_material<Ms...>, Ts...> const& surf)
    -> decltype( surf.material.is_emissive() )
{
    return surf.material.is_emissive();
}

template <typename N, typename ...Ms, typename ...Ts>
VSNRAY_FUNC
inline auto has_emissive_material(surface<N, simd::generic_material4<Ms...>, Ts...> const& surf)
    -> decltype( surf.material.is_emissive() )
{
    return surf.material.is_emissive();
}

} // visionaray

#endif // VSNRAY_SURFACE_INL
