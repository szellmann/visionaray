// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SURFACE_INL
#define VSNRAY_SURFACE_INL

#include <iterator>
#include <stdexcept>

#include "../generic_prim.h"
#include "../generic_material.h"
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

template <template <typename, typename...> class M, typename ...Args>
inline surface<M<simd::float8>> pack(
        surface<M<float, Args...>> const& s1,
        surface<M<float, Args...>> const& s2,
        surface<M<float, Args...>> const& s3,
        surface<M<float, Args...>> const& s4,
        surface<M<float, Args...>> const& s5,
        surface<M<float, Args...>> const& s6,
        surface<M<float, Args...>> const& s7,
        surface<M<float, Args...>> const& s8
        )
{
    return surface<M<simd::float8, Args...>>(
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

template <template <typename, typename...> class M, typename C, typename ...Args>
inline surface<M<simd::float8>, vector<3, simd::float8>> pack(
        surface<M<float, Args...>, C> const& s1,
        surface<M<float, Args...>, C> const& s2,
        surface<M<float, Args...>, C> const& s3,
        surface<M<float, Args...>, C> const& s4,
        surface<M<float, Args...>, C> const& s5,
        surface<M<float, Args...>, C> const& s6,
        surface<M<float, Args...>, C> const& s7,
        surface<M<float, Args...>, C> const& s8
        )
{
    return surface<M<simd::float8, Args...>, vector<3, simd::float8>>(
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
    auto n1 = normals[hr.prim_id * 3];
    auto n2 = normals[hr.prim_id * 3 + 1];
    auto n3 = normals[hr.prim_id * 3 + 2];

    auto s2 = n3 * hr.v;
    auto s3 = n2 * hr.u;
    auto s1 = n1 * (1.0f - (hr.u + hr.v));

    return normalize(s1 + s2 + s3);
}


//-------------------------------------------------------------------------------------------------
// Get texture coordinate from array
//

template <typename TexCoords, typename HR>
VSNRAY_FUNC
inline auto get_tex_coord(TexCoords tex_coords, HR const& hr)
    -> typename std::iterator_traits<TexCoords>::value_type
{
    auto tc1 = tex_coords[hr.prim_id * 3];
    auto tc2 = tex_coords[hr.prim_id * 3 + 1];
    auto tc3 = tex_coords[hr.prim_id * 3 + 2];

    auto s2 = tc3 * hr.v;
    auto s3 = tc2 * hr.u;
    auto s1 = tc1 * (1.0f - (hr.u + hr.v));

    return s1 + s2 + s3;
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
    auto hr4 = simd::unpack(hr);

    auto off1 = hr4[0].prim_id * 3;
    auto off2 = hr4[1].prim_id * 3;
    auto off3 = hr4[2].prim_id * 3;
    auto off4 = hr4[3].prim_id * 3;

    vector<2, simd::float4> tc1(
            simd::float4( coords[off1    ].x, coords[off2    ].x, coords[off3    ].x, coords[off4    ].x ),
            simd::float4( coords[off1    ].y, coords[off2    ].y, coords[off3    ].y, coords[off4    ].y )
            );
                
    vector<2, simd::float4> tc2(
            simd::float4( coords[off1 + 1].x, coords[off2 + 1].x, coords[off3 + 1].x, coords[off4 + 1].x ),
            simd::float4( coords[off1 + 1].y, coords[off2 + 1].y, coords[off3 + 1].y, coords[off4 + 1].y )
            );

    vector<2, simd::float4> tc3(
            simd::float4( coords[off1 + 2].x, coords[off2 + 2].x, coords[off3 + 2].x, coords[off4 + 2].x ),
            simd::float4( coords[off1 + 2].y, coords[off2 + 2].y, coords[off3 + 2].y, coords[off4 + 2].y )
            );

    auto s2 = tc3 * hr.v;
    auto s3 = tc2 * hr.u;
    auto s1 = tc1 * (simd::float4(1.0) - (hr.u + hr.v));
    return s1 + s2 + s3;
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
inline auto get_surface_any_prim(
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
inline auto get_surface_any_prim(
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

template <typename R, typename NormalBinding, typename Normals, typename Materials>
VSNRAY_FUNC
inline auto get_surface_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        basic_triangle<3, float> const*             primitives,
        Normals                                     normals,
        Materials                                   materials,
        NormalBinding                               /* */
        )
    -> surface<typename std::iterator_traits<Materials>::value_type>
{
    VSNRAY_UNUSED(primitives);
    return get_surface_impl(hr, normals, materials, NormalBinding());
}


//-------------------------------------------------------------------------------------------------
// Generic primitive / float
//

template <typename R, typename NormalBinding, typename Normals, typename Materials>
VSNRAY_FUNC
inline auto get_surface_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        generic_prim const*                         primitives,
        Normals                                     normals,
        Materials                                   materials,
        NormalBinding                               /* */
        )
    -> surface<typename std::iterator_traits<Materials>::value_type>
{
    using M = typename std::iterator_traits<Materials>::value_type;

    vector<3, float> n;
    switch (hr.prim_type)
    {

    case detail::TrianglePrimitive:
        n = normals[hr.prim_id];
        break;

    case detail::SpherePrimitive:
    {
        basic_sphere<float> sphere = primitives[hr.prim_id].sphere;
        n = (hr.isect_pos - sphere.center) / sphere.radius;
        break;
    }

    default:
        throw std::runtime_error("primitive type unspecified");

    }

    return surface<M>( n, materials[hr.geom_id] );
}


//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / float4
//

template <typename NormalBinding, typename Normals, typename Materials>
inline auto get_surface_any_prim(
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
inline auto get_surface_any_prim(
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
    typename Normals,
    typename Materials
    >
VSNRAY_FUNC
inline auto get_surface_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        B<basic_triangle<3, float>> const*          tree,
        Normals                                     normals,
        Materials                                   materials,
        NormalBinding                               /* */
        ) -> decltype( get_surface_any_prim(
                hr,
                normals,
                materials,
                NormalBinding()
                ) )
{
    VSNRAY_UNUSED(tree);
    return get_surface_any_prim(hr, normals, materials, NormalBinding());
}

template <
    typename R,
    typename NormalBinding,
    template <typename> class B,
    typename Normals,
    typename TexCoords,
    typename Materials,
    typename Textures
    >
VSNRAY_FUNC
inline auto get_surface_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        B<basic_triangle<3, float>> const*          tree,
        Normals                                     normals,
        TexCoords                                   tex_coords,
        Materials                                   materials,
        Textures                                    textures,
        NormalBinding                               /* */
        ) -> decltype( get_surface_any_prim(
                hr,
                normals,
                tex_coords,
                materials,
                textures,
                NormalBinding()
                ) )
{
    VSNRAY_UNUSED(tree);
    return get_surface_any_prim(
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
inline auto get_surface_impl(
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        basic_triangle<3, float> const*                     primitives,
        Normals                                             normals,
        Materials                                           materials,
        NormalBinding                                       /* */
        ) -> decltype( get_surface_impl(hr, normals, materials, NormalBinding()) )
{
    VSNRAY_UNUSED(primitives);
    return get_surface_any_prim(hr, normals, materials, NormalBinding());
}


//-------------------------------------------------------------------------------------------------
// Generic primitive / float4
//

template <typename NormalBinding, unsigned C, typename Normals>
inline auto simd_normal
(
    hit_record<simd::ray4, primitive<unsigned>> const&  hr4,
    hit_record<ray, primitive<unsigned>> const&         hr,
    generic_prim const*                                 primitives,
    Normals                                             normals,
    NormalBinding                                       /* */
) -> typename std::iterator_traits<Normals>::value_type
{
    using N = typename std::iterator_traits<Normals>::value_type;

    switch (hr.prim_type)
    {

    case detail::TrianglePrimitive:
        return get_normal(normals, hr, NormalBinding());

    case detail::SpherePrimitive:
    {
        basic_sphere<float> sphere = primitives[hr.prim_id].sphere;

        auto tmp = hr4.isect_pos;
        N isect_pos
        (
            _mm_cvtss_f32(simd::shuffle<C, 0, 0, 0>(tmp.x)),
            _mm_cvtss_f32(simd::shuffle<C, 0, 0, 0>(tmp.y)),
            _mm_cvtss_f32(simd::shuffle<C, 0, 0, 0>(tmp.z))
        );

        return (isect_pos - sphere.center) / sphere.radius;

    }

    default:
         throw std::runtime_error("primitive type unspecified");

    }
}

template <typename NormalBinding, typename Normals, typename Materials>
inline auto get_surface_impl(
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        generic_prim const*                                 primitives,
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
                hr4[0].hit ? simd_normal<0>(hr, hr4[0], primitives, normals, NormalBinding()) : N(),
                hr4[0].hit ? materials[hr4[0].geom_id]                                        : M()
                ),
            surface<M>(
                hr4[1].hit ? simd_normal<1>(hr, hr4[1], primitives, normals, NormalBinding()) : N(),
                hr4[1].hit ? materials[hr4[1].geom_id]                                        : M()
                ),
            surface<M>(
                hr4[2].hit ? simd_normal<2>(hr, hr4[2], primitives, normals, NormalBinding()) : N(),
                hr4[2].hit ? materials[hr4[2].geom_id]                                        : M()
                ),
            surface<M>(
                hr4[3].hit ? simd_normal<3>(hr, hr4[3], primitives, normals, NormalBinding()) : N(),
                hr4[3].hit ? materials[hr4[3].geom_id]                                        : M()
                )
            );
}


//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / float8
//

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename NormalBinding, typename Normals, typename Materials>
inline auto get_surface_any_prim(
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

template <typename NormalBinding, typename Normals, typename Materials>
inline auto get_surface_impl(
        hit_record<simd::ray8, primitive<unsigned>> const&  hr,
        basic_triangle<3, float> const*                     primitives,
        Normals                                             normals,
        Materials                                           materials,
        NormalBinding                                       /* */
        ) -> decltype( get_surface_any_prim(
                hr,
                normals,
                materials,
                NormalBinding()
                ) )
{
    VSNRAY_UNUSED(primitives);
    return get_surface_any_prim(hr, normals, materials, NormalBinding());
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


//-------------------------------------------------------------------------------------------------
// Functions to deduce appropriate surface via parameter inspection
//

template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface_impl(HR const& hr, Params const& p, detail::has_no_textures_tag)
    -> decltype( get_surface_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.materials,
            typename Params::normal_binding{}
            ) )
{
    return get_surface_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.materials,
            typename Params::normal_binding{}
            );
}

template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface_impl(HR const& hr, Params const& p, detail::has_textures_tag)
    -> decltype( get_surface_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.tex_coords,
            p.materials,
            p.textures,
            typename Params::normal_binding{}
            ) )
{
    return get_surface_impl(
            hr,
            p.prims.begin,
            p.normals,
            p.tex_coords,
            p.materials,
            p.textures,
            typename Params::normal_binding{}
            );
}

template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface(HR const& hr, Params const& p)
    -> decltype( get_surface_impl(
            hr,
            p,
            detail::has_textures<Params>{}
            ) )
{
    return get_surface_impl(
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
