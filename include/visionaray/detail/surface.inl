// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SURFACE_INL
#define VSNRAY_SURFACE_INL 1

#include <iterator>
#include <stdexcept>

#include "../generic_material.h"
#include "../generic_primitive.h"
#include "../tags.h"

namespace visionaray
{
namespace simd
{

namespace detail
{

template <typename M, typename N>
surface<M> make_surface(N const& n, M const& m)
{
    return surface<M>(n, m);
}

template <typename M, typename C, typename N>
surface<M, C> make_surface(N const& n, M const m, C const& tex_color)
{
    return surface<M, C>(n, m, tex_color);
}

} // detail

//-------------------------------------------------------------------------------------------------
// Functions to pack and unpack SIMD surfaces
//

template <typename M, typename ...Args>
inline auto pack(
        surface<M, Args...> const& s1,
        surface<M, Args...> const& s2,
        surface<M, Args...> const& s3,
        surface<M, Args...> const& s4
        ) -> decltype( detail::make_surface(
            pack(
                s1.normal,
                s2.normal,
                s3.normal,
                s4.normal
                ),
            pack(
                s1.material,
                s2.material,
                s3.material,
                s4.material
                )
            ) )

{
    return detail::make_surface(
            pack(
                s1.normal,
                s2.normal,
                s3.normal,
                s4.normal
                ),
            pack(
                s1.material,
                s2.material,
                s3.material,
                s4.material
                )
            );
}

template <typename M, typename C, typename ...Args>
inline auto pack(
        surface<M, C, Args...> const& s1,
        surface<M, C, Args...> const& s2,
        surface<M, C, Args...> const& s3,
        surface<M, C, Args...> const& s4
        ) -> decltype( detail::make_surface(
            pack(
                s1.normal,
                s2.normal,
                s3.normal,
                s4.normal
                ),
            pack(
                s1.material,
                s2.material,
                s3.material,
                s4.material
                ),
            pack(
                s1.tex_color_,
                s2.tex_color_,
                s3.tex_color_,
                s4.tex_color_
                )
            ) )
{
    return detail::make_surface(
            pack(
                s1.normal,
                s2.normal,
                s3.normal,
                s4.normal
                ),
            pack(
                s1.material,
                s2.material,
                s3.material,
                s4.material
                ),
            pack(
                s1.tex_color_,
                s2.tex_color_,
                s3.tex_color_,
                s4.tex_color_
                )
            );
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename M, typename ...Args>
inline auto pack(
        surface<M, Args...> const& s1,
        surface<M, Args...> const& s2,
        surface<M, Args...> const& s3,
        surface<M, Args...> const& s4,
        surface<M, Args...> const& s5,
        surface<M, Args...> const& s6,
        surface<M, Args...> const& s7,
        surface<M, Args...> const& s8
        ) -> decltype( detail::make_surface(
            pack(
                s1.normal,
                s2.normal,
                s3.normal,
                s4.normal,
                s5.normal,
                s6.normal,
                s7.normal,
                s8.normal
                ),
            pack(
                s1.material,
                s2.material,
                s3.material,
                s4.material,
                s5.material,
                s6.material,
                s7.material,
                s8.material
                )
            ) )
{
    return detail::make_surface(
            pack(
                s1.normal,
                s2.normal,
                s3.normal,
                s4.normal,
                s5.normal,
                s6.normal,
                s7.normal,
                s8.normal
                ),
            pack(
                s1.material,
                s2.material,
                s3.material,
                s4.material,
                s5.material,
                s6.material,
                s7.material,
                s8.material
                )
            );
}

template <typename M, typename C, typename ...Args>
inline auto pack(
        surface<M, C, Args...> const& s1,
        surface<M, C, Args...> const& s2,
        surface<M, C, Args...> const& s3,
        surface<M, C, Args...> const& s4,
        surface<M, C, Args...> const& s5,
        surface<M, C, Args...> const& s6,
        surface<M, C, Args...> const& s7,
        surface<M, C, Args...> const& s8
        ) -> decltype( detail::make_surface(
            pack(
                s1.normal,
                s2.normal,
                s3.normal,
                s4.normal,
                s5.normal,
                s6.normal,
                s7.normal,
                s8.normal
                ),
            pack(
                s1.material,
                s2.material,
                s3.material,
                s4.material,
                s5.material,
                s6.material,
                s7.material,
                s8.material
                ),
            pack(
                s1.tex_color_,
                s2.tex_color_,
                s3.tex_color_,
                s4.tex_color_,
                s5.tex_color_,
                s6.tex_color_,
                s7.tex_color_,
                s8.tex_color_
                )
            ) )
{
    return detail::make_surface(
            pack(
                s1.normal,
                s2.normal,
                s3.normal,
                s4.normal,
                s5.normal,
                s6.normal,
                s7.normal,
                s8.normal
                ),
            pack(
                s1.material,
                s2.material,
                s3.material,
                s4.material,
                s5.material,
                s6.material,
                s7.material,
                s8.material
                ),
            pack(
                s1.tex_color_,
                s2.tex_color_,
                s3.tex_color_,
                s4.tex_color_,
                s5.tex_color_,
                s6.tex_color_,
                s7.tex_color_,
                s8.tex_color_
                )
            );
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

} // simd


//-------------------------------------------------------------------------------------------------
// Get face normal from array
//

template <typename Normals, typename HR>
VSNRAY_FUNC
inline auto get_normal(Normals normals, HR const& hr, normals_per_face_binding)
    -> typename std::iterator_traits<Normals>::value_type
{
    return normals[hr.prim_id];
}


//-------------------------------------------------------------------------------------------------
// Get vertex normal from array
//

template <typename Normals, typename HR>
VSNRAY_FUNC
inline auto get_normal(Normals normals, HR const& hr, normals_per_vertex_binding)
    -> typename std::iterator_traits<Normals>::value_type
{
    return normalize( lerp(
            normals[hr.prim_id * 3],
            normals[hr.prim_id * 3 + 1],
            normals[hr.prim_id * 3 + 2],
            hr.u,
            hr.v
            ) );
}


//-------------------------------------------------------------------------------------------------
// Gather four face normals with SSE
//

template <typename Normals>
inline vector<3, simd::float4> get_normal(
        Normals                                             normals,
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        normals_per_face_binding
        )
{
    using N = typename std::iterator_traits<Normals>::value_type;

    auto hr4 = simd::unpack(hr);

    auto get_norm = [&](int x)
    {
        return hr4[x].hit ? normals[hr4[x].prim_id] : N();
    };

    auto n1 = get_norm(0);
    auto n2 = get_norm(1);
    auto n3 = get_norm(2);
    auto n4 = get_norm(3);

    return vector<3, simd::float4>(
            simd::float4( n1.x, n2.x, n3.x, n4.x ),
            simd::float4( n1.y, n2.y, n3.y, n4.y ),
            simd::float4( n1.z, n2.z, n3.z, n4.z )
            );
}


//-------------------------------------------------------------------------------------------------
// Gather four vertex normals with SSE
//

template <typename Normals>
inline vector<3, simd::float4> get_normal(
        Normals                                             normals,
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        normals_per_vertex_binding
        )
{
    using N = typename std::iterator_traits<Normals>::value_type;

    auto hr4 = simd::unpack(hr);

    auto get_norm = [&](int x, int y)
    {
        return hr4[x].hit ? normals[hr4[x].prim_id * 3 + y] : N();
    };

    vector<3, simd::float4> n1(
            simd::float4( get_norm(0, 0).x, get_norm(1, 0).x, get_norm(2, 0).x, get_norm(3, 0).x ),
            simd::float4( get_norm(0, 0).y, get_norm(1, 0).y, get_norm(2, 0).y, get_norm(3, 0).y ),
            simd::float4( get_norm(0, 0).z, get_norm(1, 0).z, get_norm(2, 0).z, get_norm(3, 0).z )
            );

    vector<3, simd::float4> n2(
            simd::float4( get_norm(0, 1).x, get_norm(1, 1).x, get_norm(2, 1).x, get_norm(3, 1).x ),
            simd::float4( get_norm(0, 1).y, get_norm(1, 1).y, get_norm(2, 1).y, get_norm(3, 1).y ),
            simd::float4( get_norm(0, 1).z, get_norm(1, 1).z, get_norm(2, 1).z, get_norm(3, 1).z )
            );

    vector<3, simd::float4> n3(
            simd::float4( get_norm(0, 2).x, get_norm(1, 2).x, get_norm(2, 2).x, get_norm(3, 2).x ),
            simd::float4( get_norm(0, 2).y, get_norm(1, 2).y, get_norm(2, 2).y, get_norm(3, 2).y ),
            simd::float4( get_norm(0, 2).z, get_norm(1, 2).z, get_norm(2, 2).z, get_norm(3, 2).z )
            );

    return normalize( lerp(n1, n2, n3, hr.u, hr.v) );
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// Gather eight face normals with AVX
//

template <typename Normals>
inline vector<3, simd::float8> get_normal(
        Normals                                             normals,
        hit_record<simd::ray8, primitive<unsigned>> const&  hr,
        normals_per_face_binding
        )
{
    using N = typename std::iterator_traits<Normals>::value_type;

    auto hr8 = simd::unpack(hr);

    auto get_norm = [&](int x)
    {
        return hr8[x].hit ? normals[hr8[x].prim_id] : N();
    };

    auto n1 = get_norm(0);
    auto n2 = get_norm(1);
    auto n3 = get_norm(2);
    auto n4 = get_norm(3);
    auto n5 = get_norm(4);
    auto n6 = get_norm(5);
    auto n7 = get_norm(6);
    auto n8 = get_norm(7);

    return vector<3, simd::float8>(
            simd::float8( n1.x, n2.x, n3.x, n4.x, n5.x, n6.x, n7.x, n8.x ),
            simd::float8( n1.y, n2.y, n3.y, n4.y, n5.y, n6.y, n7.y, n8.y ),
            simd::float8( n1.z, n2.z, n3.z, n4.z, n5.z, n6.z, n7.z, n8.z )
            );
}


//-------------------------------------------------------------------------------------------------
// Gather eight vertex normals with AVX
//

template <typename Normals>
inline vector<3, simd::float8> get_normal(
        Normals                                             normals,
        hit_record<simd::ray8, primitive<unsigned>> const&  hr,
        normals_per_vertex_binding
        )
{
    using N = typename std::iterator_traits<Normals>::value_type;

    auto hr8 = simd::unpack(hr);

    auto get_norm = [&](int x, int y)
    {
        return hr8[x].hit ? normals[hr8[x].prim_id * 3 + y] : N();
    };

    vector<3, simd::float8> n1(
            simd::float8( get_norm(0, 0).x, get_norm(1, 0).x, get_norm(2, 0).x, get_norm(3, 0).x,
                          get_norm(4, 0).x, get_norm(5, 0).x, get_norm(6, 0).x, get_norm(7, 0).x ),
            simd::float8( get_norm(0, 0).y, get_norm(1, 0).y, get_norm(2, 0).y, get_norm(3, 0).y,
                          get_norm(4, 0).y, get_norm(5, 0).y, get_norm(6, 0).y, get_norm(7, 0).y ),
            simd::float8( get_norm(0, 0).z, get_norm(1, 0).z, get_norm(2, 0).z, get_norm(3, 0).z,
                          get_norm(4, 0).z, get_norm(5, 0).z, get_norm(6, 0).z, get_norm(7, 0).z )
            );

    vector<3, simd::float8> n2(
            simd::float8( get_norm(0, 1).x, get_norm(1, 1).x, get_norm(2, 1).x, get_norm(3, 1).x,
                          get_norm(4, 1).x, get_norm(5, 1).x, get_norm(6, 1).x, get_norm(7, 1).x ),
            simd::float8( get_norm(0, 1).y, get_norm(1, 1).y, get_norm(2, 1).y, get_norm(3, 1).y,
                          get_norm(4, 1).y, get_norm(5, 1).y, get_norm(6, 1).y, get_norm(7, 1).y ),
            simd::float8( get_norm(0, 1).z, get_norm(1, 1).z, get_norm(2, 1).z, get_norm(3, 1).z,
                          get_norm(4, 1).z, get_norm(5, 1).z, get_norm(6, 1).z, get_norm(7, 1).z )
            );

    vector<3, simd::float8> n3(
            simd::float8( get_norm(0, 2).x, get_norm(1, 2).x, get_norm(2, 2).x, get_norm(3, 2).x,
                          get_norm(4, 2).x, get_norm(5, 2).x, get_norm(6, 2).x, get_norm(7, 2).x ),
            simd::float8( get_norm(0, 2).y, get_norm(1, 2).y, get_norm(2, 2).y, get_norm(3, 2).y,
                          get_norm(4, 2).y, get_norm(5, 2).y, get_norm(6, 2).y, get_norm(7, 2).y ),
            simd::float8( get_norm(0, 2).z, get_norm(1, 2).z, get_norm(2, 2).z, get_norm(3, 2).z,
                          get_norm(4, 2).z, get_norm(5, 2).z, get_norm(6, 2).z, get_norm(7, 2).z )
            );

    return normalize( lerp(n1, n2, n3, hr.u, hr.v) );
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


//-------------------------------------------------------------------------------------------------
// Get texture coordinate from array
//

template <typename TexCoords, typename HR>
VSNRAY_FUNC
inline auto get_tex_coord(TexCoords tex_coords, HR const& hr)
    -> typename std::iterator_traits<TexCoords>::value_type
{
    return lerp(
            tex_coords[hr.prim_id * 3],
            tex_coords[hr.prim_id * 3 + 1],
            tex_coords[hr.prim_id * 3 + 2],
            hr.u,
            hr.v
            );
}


//-------------------------------------------------------------------------------------------------
// Gather four texture coordinates with SSE
//

template <typename TexCoords>
inline vector<2, simd::float4> get_tex_coord(
        TexCoords                                           coords,
        hit_record<simd::ray4, primitive<unsigned>> const&  hr
        )
{
    using TC = typename std::iterator_traits<TexCoords>::value_type;

    auto hr4 = simd::unpack(hr);

    auto get_coord = [&](int x, int y)
    {
        return hr4[x].hit ? coords[hr4[x].prim_id * 3 + y] : TC();
    };

    vector<2, simd::float4> tc1(
            simd::float4( get_coord(0, 0).x, get_coord(1, 0).x, get_coord(2, 0).x, get_coord(3, 0).x ),
            simd::float4( get_coord(0, 0).y, get_coord(1, 0).y, get_coord(2, 0).y, get_coord(3, 0).y )
            );
                
    vector<2, simd::float4> tc2(
            simd::float4( get_coord(0, 1).x, get_coord(1, 1).x, get_coord(2, 1).x, get_coord(3, 1).x ),
            simd::float4( get_coord(0, 1).y, get_coord(1, 1).y, get_coord(2, 1).y, get_coord(3, 1).y )
            );

    vector<2, simd::float4> tc3(
            simd::float4( get_coord(0, 2).x, get_coord(1, 2).x, get_coord(2, 2).x, get_coord(3, 2).x ),
            simd::float4( get_coord(0, 2).y, get_coord(1, 2).y, get_coord(2, 2).y, get_coord(3, 2).y )
            );

    return lerp( tc1, tc2, tc3, hr.u, hr.v );
}


//-------------------------------------------------------------------------------------------------
// Gather four texture coordinates from array
//

template <typename TexCoords, typename HR>
inline auto get_tex_coord(TexCoords coords, std::array<HR, 4> const& hr)
    -> std::array<typename std::iterator_traits<TexCoords>::value_type, 4>
{
    using TC = typename std::iterator_traits<TexCoords>::value_type;

    return std::array<TC, 4>
    {{
        get_tex_coord(coords, hr[0]),
        get_tex_coord(coords, hr[1]),
        get_tex_coord(coords, hr[2]),
        get_tex_coord(coords, hr[3])
    }};
}


//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / float
//

template <typename R, typename NormalBinding, typename Normals, typename Materials>
VSNRAY_FUNC
inline auto get_surface_any_prim_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        Normals                                     normals,
        Materials                                   materials,
        NormalBinding                               /* */
        )
    -> surface<typename std::iterator_traits<Materials>::value_type>
{
    using M = typename std::iterator_traits<Materials>::value_type;

    return surface<M>(
            get_normal(normals, hr, NormalBinding()),
            materials[hr.geom_id]
            );
}

template <
    typename R,
    typename NormalBinding,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Textures
    >
VSNRAY_FUNC
inline auto get_surface_any_prim_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        Normals                                     normals,
        TexCoords                                   tex_coords,
        Materials                                   materials,
        Textures                                    textures,
        NormalBinding                               /* */
        )
    -> surface<typename std::iterator_traits<Materials>::value_type, vector<3, float>>
{
    using M = typename std::iterator_traits<Materials>::value_type;

    auto tc = get_tex_coord(tex_coords, hr);

    auto const& tex = textures[hr.geom_id];
    auto tex_color = tex.width() > 0 && tex.height() > 0
                   ? vector<3, float>(tex2D(tex, tc))
                   : vector<3, float>(1.0);

    auto normal = get_normal(normals, hr, NormalBinding());
    return surface<M, vector<3, float>>( normal, materials[hr.geom_id], tex_color );
}


//-------------------------------------------------------------------------------------------------
// Triangle / float
//

template <
    typename R,
    typename NormalBinding,
    typename Primitives,
    typename Normals,
    typename Materials
    >
VSNRAY_FUNC
inline auto get_surface_with_prims_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        Primitives                                  primitives,
        Normals                                     normals,
        Materials                                   materials,
        basic_triangle<3, float>                    /* */,
        NormalBinding                               /* */
        )
    -> decltype( get_surface_any_prim_impl(hr, normals, materials, NormalBinding()) )
{
    VSNRAY_UNUSED(primitives);
    return get_surface_any_prim_impl(hr, normals, materials, NormalBinding());
}


//-------------------------------------------------------------------------------------------------
// Generic primitive / float
//

namespace detail
{

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

    VSNRAY_FUNC
    return_type operator()(basic_sphere<float> const& sphere) const
    {
        return (hr_.isect_pos - sphere.center) / sphere.radius; // TODO: (custom) get_normal() functions?
    }

    template <typename Primitive>
    VSNRAY_FUNC
    return_type operator()(Primitive const& primitive) const
    {
        VSNRAY_UNUSED(primitive);

        return get_normal(normals_, hr_, NormalBinding());
    }

private:

    Normals     normals_;
    HR const&   hr_;

};

} // detail

template <
    typename R,
    typename NormalBinding,
    typename Primitives,
    typename Normals,
    typename Materials,
    typename ...Ts
    >
VSNRAY_FUNC
inline auto get_surface_with_prims_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        Primitives                                  primitives,
        Normals                                     normals,
        Materials                                   materials,
        generic_primitive<Ts...>                    /* */,
        NormalBinding                               /* */
        )
    -> surface<typename std::iterator_traits<Materials>::value_type>
{
    using M = typename std::iterator_traits<Materials>::value_type;
    using HR = hit_record<R, primitive<unsigned>>;

    detail::get_normal_from_generic_primitive_visitor<NormalBinding, Normals, HR> visitor(
            normals,
            hr
            );

    auto n = apply_visitor( visitor, primitives[hr.prim_id] );
    return surface<M>( n, materials[hr.geom_id] );
}


//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / float4
//

template <typename NormalBinding, typename Normals, typename Materials>
inline auto get_surface_any_prim_impl(
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        Normals                                             normals,
        Materials                                           materials,
        NormalBinding                                       /* */
        ) -> decltype( simd::pack(
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>()
                ) )
{
    using N = typename std::iterator_traits<Normals>::value_type;
    using M = typename std::iterator_traits<Materials>::value_type;

    auto hr4 = simd::unpack(hr);

    return simd::pack(
            surface<M>(
                hr4[0].hit ? get_normal(normals, hr4[0], NormalBinding()) : N(),
                hr4[0].hit ? materials[hr4[0].geom_id]                    : M()
                ),
            surface<M>(
                hr4[1].hit ? get_normal(normals, hr4[1], NormalBinding()) : N(),
                hr4[1].hit ? materials[hr4[1].geom_id]                    : M()
                ),
            surface<M>(
                hr4[2].hit ? get_normal(normals, hr4[2], NormalBinding()) : N(),
                hr4[2].hit ? materials[hr4[2].geom_id]                    : M()
                ),
            surface<M>(
                hr4[3].hit ? get_normal(normals, hr4[3], NormalBinding()) : N(),
                hr4[3].hit ? materials[hr4[3].geom_id]                    : M()
                )
            );
}


template <
    typename NormalBinding,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Textures
    >
inline auto get_surface_any_prim_impl(
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        Normals                                             normals,
        TexCoords                                           tex_coords,
        Materials                                           materials,
        Textures                                            textures,
        NormalBinding                                       /* */
        ) -> decltype( simd::pack(
                surface<typename std::iterator_traits<Materials>::value_type, vector<3, float>>(),
                surface<typename std::iterator_traits<Materials>::value_type, vector<3, float>>(),
                surface<typename std::iterator_traits<Materials>::value_type, vector<3, float>>(),
                surface<typename std::iterator_traits<Materials>::value_type, vector<3, float>>()
                ) )
{
    using N = typename std::iterator_traits<Normals>::value_type;
    using M = typename std::iterator_traits<Materials>::value_type;

    auto hr4 = simd::unpack(hr);

    auto tc4 = get_tex_coord(tex_coords, hr4);

    vector<3, float> tex_color4[4];
    for (unsigned i = 0; i < 4; ++i)
    {
        if (!hr4[i].hit)
        {
            continue;
        }

        auto const& tex = textures[hr4[i].geom_id];
        tex_color4[i] = tex.width() > 0 && tex.height() > 0
                      ? vector<3, float>(tex2D(tex, tc4[i]))
                      : vector<3, float>(1.0);
    }

    return simd::pack(
            surface<M, vector<3, float>>(
                hr4[0].hit ? get_normal(normals, hr4[0], NormalBinding()) : N(),
                hr4[0].hit ? materials[hr4[0].geom_id]                    : M(),
                hr4[0].hit ? tex_color4[0]                                : vector<3, float>(1.0)
                ),
            surface<M, vector<3, float>>(
                hr4[1].hit ? get_normal(normals, hr4[1], NormalBinding()) : N(),
                hr4[1].hit ? materials[hr4[1].geom_id]                    : M(),
                hr4[1].hit ? tex_color4[1]                                : vector<3, float>(1.0)
                ),
            surface<M, vector<3, float>>(
                hr4[2].hit ? get_normal(normals, hr4[2], NormalBinding()) : N(),
                hr4[2].hit ? materials[hr4[2].geom_id]                    : M(),
                hr4[2].hit ? tex_color4[2]                                : vector<3, float>(1.0)
                ),
            surface<M, vector<3, float>>(
                hr4[3].hit ? get_normal(normals, hr4[3], NormalBinding()) : N(),
                hr4[3].hit ? materials[hr4[3].geom_id]                    : M(),
                hr4[3].hit ? tex_color4[3]                                : vector<3, float>(1.0)
                )
            );
}


//-------------------------------------------------------------------------------------------------
// Bvh / float|float4|float8|...
//

template <
    typename R,
    typename NormalBinding,
    template <typename> class B,
    typename Primitives,
    typename Normals,
    typename Materials
    >
VSNRAY_FUNC
inline auto get_surface_with_prims_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        Primitives                                  primitives,
        Normals                                     normals,
        Materials                                   materials,
        B<basic_triangle<3, float>>                 /* */,
        NormalBinding                               /* */
        ) -> decltype( get_surface_any_prim_impl(
                hr,
                normals,
                materials,
                NormalBinding()
                ) )
{
    VSNRAY_UNUSED(primitives);
    return get_surface_any_prim_impl(hr, normals, materials, NormalBinding());
}

template <
    typename R,
    typename NormalBinding,
    template <typename> class B,
    typename Primitives,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Textures
    >
VSNRAY_FUNC
inline auto get_surface_with_prims_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        Primitives                                  primitives,
        Normals                                     normals,
        TexCoords                                   tex_coords,
        Materials                                   materials,
        Textures                                    textures,
        B<basic_triangle<3, float>>                 /* */,
        NormalBinding                               /* */
        ) -> decltype( get_surface_any_prim_impl(
                hr,
                normals,
                tex_coords,
                materials,
                textures,
                NormalBinding()
                ) )
{
    VSNRAY_UNUSED(primitives);
    return get_surface_any_prim_impl(
            hr,
            normals,
            tex_coords,
            materials,
            textures,
            NormalBinding()
            );
}


//-------------------------------------------------------------------------------------------------
// Triangle / float4
//

template <typename NormalBinding, typename Normals, typename Materials>
inline auto get_surface_with_prims_impl(
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        basic_triangle<3, float> const*                     primitives,
        Normals                                             normals,
        Materials                                           materials,
        basic_triangle<3, float>                            /* */,
        NormalBinding                                       /* */
        ) -> decltype( get_surface_any_prim_impl(hr, normals, materials, NormalBinding()) )
{
    VSNRAY_UNUSED(primitives);
    return get_surface_any_prim_impl(hr, normals, materials, NormalBinding());
}

template <
    typename NormalBinding,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Textures
    >
inline auto get_surface_with_prims_impl(
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        basic_triangle<3, float> const*                     primitives,
        Normals                                             normals,
        TexCoords                                           tex_coords,
        Materials                                           materials,
        Textures                                            textures,
        basic_triangle<3, float>                            /* */,
        NormalBinding                                       /* */
        ) -> decltype( get_surface_any_prim_impl(
                hr,
                normals,
                tex_coords,
                materials,
                textures,
                NormalBinding())
                )
{
    VSNRAY_UNUSED(primitives);
    return get_surface_any_prim_impl(
            hr,
            normals,
            tex_coords,
            materials,
            textures,
            NormalBinding()
            );
}


//-------------------------------------------------------------------------------------------------
// Generic primitive / float4
//

template <
    typename NormalBinding,
    typename Primitives,
    typename Normals,
    typename Materials,
    typename ...Ts>
inline auto get_surface_with_prims_impl(
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        Primitives                                          primitives,
        Normals                                             normals,
        Materials                                           materials,
        generic_primitive<Ts...>                            /* */,
        NormalBinding                                       /* */
        ) -> decltype( simd::pack(
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>()
                ) )
{
    using N = typename std::iterator_traits<Normals>::value_type;
    using M = typename std::iterator_traits<Materials>::value_type;

    auto hr4 = simd::unpack(hr); 

    auto get_surf = [&](unsigned index)
    {
        // dispatch to scalar version of this function
        return get_surface_with_prims_impl(
                hr4[index],
                primitives,
                normals,
                materials,
                generic_primitive<Ts...>(),
                NormalBinding()
                );
    };

    return simd::pack(
            hr4[0].hit ? get_surf(0) : surface<M>( N(), M() ),
            hr4[1].hit ? get_surf(1) : surface<M>( N(), M() ),
            hr4[2].hit ? get_surf(2) : surface<M>( N(), M() ),
            hr4[3].hit ? get_surf(3) : surface<M>( N(), M() )
            );
}


//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / float8
//

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename NormalBinding, typename Normals, typename Materials>
inline auto get_surface_any_prim_impl(
        hit_record<simd::ray8, primitive<unsigned>> const&  hr,
        Normals                                             normals,
        Materials                                           materials,
        NormalBinding                                       /* */
        ) -> decltype( simd::pack(
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>(),
                surface<typename std::iterator_traits<Materials>::value_type>()
                ) )
{
    using N = typename std::iterator_traits<Normals>::value_type;
    using M = typename std::iterator_traits<Materials>::value_type;

    auto hr8 = simd::unpack(hr);

    return simd::pack(
            surface<M>(
                hr8[0].hit ? get_normal(normals, hr8[0], NormalBinding()) : N(),
                hr8[0].hit ? materials[hr8[0].geom_id]                    : M()
                ),
            surface<M>(
                hr8[1].hit ? get_normal(normals, hr8[1], NormalBinding()) : N(),
                hr8[1].hit ? materials[hr8[1].geom_id]                    : M()
                ),
            surface<M>(
                hr8[2].hit ? get_normal(normals, hr8[2], NormalBinding()) : N(),
                hr8[2].hit ? materials[hr8[2].geom_id]                    : M()
                ),
            surface<M>(
                hr8[3].hit ? get_normal(normals, hr8[3], NormalBinding()) : N(),
                hr8[3].hit ? materials[hr8[3].geom_id]                    : M()
                ),
            surface<M>(
                hr8[4].hit ? get_normal(normals, hr8[4], NormalBinding()) : N(),
                hr8[4].hit ? materials[hr8[4].geom_id]                    : M()
                ),
            surface<M>(
                hr8[5].hit ? get_normal(normals, hr8[5], NormalBinding()) : N(),
                hr8[5].hit ? materials[hr8[5].geom_id]                    : M()
                ),
            surface<M>(
                hr8[6].hit ? get_normal(normals, hr8[6], NormalBinding()) : N(),
                hr8[6].hit ? materials[hr8[6].geom_id]                    : M()
                ),
            surface<M>(
                hr8[7].hit ? get_normal(normals, hr8[7], NormalBinding()) : N(),
                hr8[7].hit ? materials[hr8[7].geom_id]                    : M()
                )
            );
}


//-------------------------------------------------------------------------------------------------
// Triangle / float8
//

template <typename NormalBinding, typename Primitives, typename Normals, typename Materials>
inline auto get_surface_with_prims_impl(
        hit_record<simd::ray8, primitive<unsigned>> const&  hr,
        Primitives                                          primitives,
        Normals                                             normals,
        Materials                                           materials,
        basic_triangle<3, float>                            /* */,
        NormalBinding                                       /* */
        ) -> decltype( get_surface_any_prim_impl(
                hr,
                normals,
                materials,
                NormalBinding()
                ) )
{
    VSNRAY_UNUSED(primitives);
    return get_surface_any_prim_impl(hr, normals, materials, NormalBinding());
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


//-------------------------------------------------------------------------------------------------
// Functions to deduce appropriate surface via parameter inspection
//

template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface_unroll_params_impl(HR const& hr, Params const& p, detail::has_no_textures_tag)
    -> decltype( get_surface_with_prims_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.materials,
            typename Params::primitive_type(),
            typename Params::normal_binding{}
            ) )
{
    return get_surface_with_prims_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.materials,
            typename Params::primitive_type(),
            typename Params::normal_binding{}
            );
}

template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface_unroll_params_impl(HR const& hr, Params const& p, detail::has_textures_tag)
    -> decltype( get_surface_with_prims_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.tex_coords,
            p.materials,
            p.textures,
            typename Params::primitive_type(),
            typename Params::normal_binding{}
            ) )
{
    return get_surface_with_prims_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.tex_coords,
            p.materials,
            p.textures,
            typename Params::primitive_type(),
            typename Params::normal_binding{}
            );
}

template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface(HR const& hr, Params const& p)
    -> decltype( get_surface_unroll_params_impl(
            hr,
            p,
            detail::has_textures<Params>{}
            ) )
{
    return get_surface_unroll_params_impl(
            hr,
            p,
            detail::has_textures<Params>{}
            );
}


//-------------------------------------------------------------------------------------------------
// General surface functions
//

template <typename M>
VSNRAY_FUNC
inline bool has_emissive_material(surface<M> const& surf)
{
    VSNRAY_UNUSED(surf);
    return false;
}

template <typename T>
VSNRAY_FUNC
inline bool has_emissive_material(surface<emissive<T>> const& surf)
{
    VSNRAY_UNUSED(surf);
    return true;
}

template <typename ...Ts>
VSNRAY_FUNC
inline auto has_emissive_material(surface<generic_material<Ts...>> const& surf)
    -> decltype( surf.material.is_emissive() )
{
    return surf.material.is_emissive();
}

template <typename ...Ts>
VSNRAY_FUNC
inline auto has_emissive_material(surface<simd::generic_material4<Ts...>> const& surf)
    -> decltype( surf.material.is_emissive() )
{
    return surf.material.is_emissive();
}

} // visionaray

#endif // VSNRAY_SURFACE_INL
