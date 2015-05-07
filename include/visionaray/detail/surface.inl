// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SURFACE_INL
#define VSNRAY_SURFACE_INL

#include <stdexcept>

#include "../generic_prim.h"
#include "../tags.h"

namespace visionaray
{

namespace simd
{

//-------------------------------------------------------------------------------------------------
// Functions to pack and unpack SIMD surfaces
//

template <template <typename, typename...> class M, typename ...Args>
inline surface<M<simd::float4>> pack(
        surface<M<float, Args...>> const& s1,
        surface<M<float, Args...>> const& s2,
        surface<M<float, Args...>> const& s3,
        surface<M<float, Args...>> const& s4
        )
{
    return surface<M<simd::float4, Args...>>(
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

template <template <typename, typename...> class M, typename C, typename ...Args>
inline surface<M<simd::float4>, vector<3, simd::float4>> pack(
        surface<M<float, Args...>, C> const& s1,
        surface<M<float, Args...>, C> const& s2,
        surface<M<float, Args...>, C> const& s3,
        surface<M<float, Args...>, C> const& s4
        )
{
    return surface<M<simd::float4, Args...>, vector<3, simd::float4>>(
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
                s1.cd_,
                s2.cd_,
                s3.cd_,
                s4.cd_
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
                s1.cd_,
                s2.cd_,
                s3.cd_,
                s4.cd_,
                s5.cd_,
                s6.cd_,
                s7.cd_,
                s8.cd_
                )
            );
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

} // simd


//-------------------------------------------------------------------------------------------------
// Get face normal from array
//

template <typename N, typename HR>
VSNRAY_FUNC
inline N get_normal(N const* normals, HR const& hr, normals_per_face_binding)
{
    return normals[hr.prim_id];
}


//-------------------------------------------------------------------------------------------------
// Get vertex normal from array
//

template <typename N, typename HR>
VSNRAY_FUNC
inline N get_normal(N const* normals, HR const& hr, normals_per_vertex_binding)
{
    N n1 = normals[hr.prim_id * 3];
    N n2 = normals[hr.prim_id * 3 + 1];
    N n3 = normals[hr.prim_id * 3 + 2];

    auto s2 = n3 * hr.v;
    auto s3 = n2 * hr.u;
    auto s1 = n1 * (1.0f - (hr.u + hr.v));

    return normalize(s1 + s2 + s3);
}


//-------------------------------------------------------------------------------------------------
// Get texture coordinate from array
//

template <typename TC, typename HR>
VSNRAY_FUNC
inline TC get_tex_coord(TC const* tex_coords, HR const& hr)
{
    TC tc1 = tex_coords[hr.prim_id * 3];
    TC tc2 = tex_coords[hr.prim_id * 3 + 1];
    TC tc3 = tex_coords[hr.prim_id * 3 + 2];

    auto s2 = tc3 * hr.v;
    auto s3 = tc2 * hr.u;
    auto s1 = tc1 * (1.0f - (hr.u + hr.v));

    return s1 + s2 + s3;
}


//-------------------------------------------------------------------------------------------------
// Gather four texture coordinates from array
//

template <typename TC, typename HR>
inline std::array<TC, 4> get_tex_coord(TC const* coords, std::array<HR, 4> const& hr)
{
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

template <typename R, typename NBinding, typename N, typename M>
VSNRAY_FUNC
inline surface<M> get_surface_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        N const*                                    normals,
        M const*                                    materials,
        NBinding                                    /* */
        )
{
    return surface<M>(
            get_normal(normals, hr, NBinding()),
            materials[hr.geom_id]
            );
}

template <typename R, typename NBinding, typename N, typename TC, typename M, typename T>
VSNRAY_FUNC
inline surface<M, vector<3, float>> get_surface_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        N const*                                    normals,
        TC const*                                   tex_coords,
        M const*                                    materials,
        T const*                                    textures,
        NBinding                                    /* */
        )
{
    auto tc  = get_tex_coord(tex_coords, hr);

    auto const& tex = textures[hr.geom_id];
    auto cd = tex.width() > 0 && tex.height() > 0 ? vector<3, float>(tex2D(tex, tc)) : vector<3, float>(1.0);

    auto normal = get_normal(normals, hr, NBinding());
    return surface<M, vector<3, float>>( normal, materials[hr.geom_id], cd );
}


//-------------------------------------------------------------------------------------------------
// Triangle / float
//

template <typename R, typename NBinding, typename N, typename M>
VSNRAY_FUNC
inline surface<M> get_surface_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        basic_triangle<3, float> const*             primitives,
        N const*                                    normals,
        M const*                                    materials,
        NBinding                                    /* */
        )
{
    VSNRAY_UNUSED(primitives);
    return get_surface_impl(hr, normals, materials, NBinding());
}


//-------------------------------------------------------------------------------------------------
// Generic primitive / float
//

template <typename R, typename NBinding, typename N, typename M>
VSNRAY_FUNC
inline surface<M> get_surface_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        generic_prim const*                         primitives,
        N const*                                    normals,
        M const*                                    materials,
        NBinding                                    /* */
        )
{

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
// Bvh / float|float4|float8|...
//

template <typename R, typename NBinding, template <typename> class B, typename N, typename M>
VSNRAY_FUNC
inline auto get_surface_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        B<basic_triangle<3, float>> const*          tree,
        N const*                                    normals,
        M const*                                    materials,
        NBinding                                    /* */
        ) -> decltype( get_surface_impl(
                hr,
                normals,
                materials,
                NBinding()
                ) )
{
    VSNRAY_UNUSED(tree);
    return get_surface_impl(hr, normals, materials, NBinding());
}

template <typename R, typename NBinding, template <typename> class B, typename N, typename TC, typename M, typename T>
VSNRAY_FUNC
inline auto get_surface_impl(
        hit_record<R, primitive<unsigned>> const&   hr,
        B<basic_triangle<3, float>> const*          tree,
        N const*                                    normals,
        TC const*                                   tex_coords,
        M const*                                    materials,
        T const*                                    textures,
        NBinding                                    /* */
        ) -> decltype( get_surface_impl(
                hr,
                normals,
                tex_coords,
                materials,
                textures,
                NBinding()
                ) )
{
    VSNRAY_UNUSED(tree);
    return get_surface_impl(
            hr,
            normals,
            tex_coords,
            materials,
            textures,
            NBinding()
            );
}


//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / float4
//

template <typename NBinding, typename N, typename M>
inline auto get_surface_impl(
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        N const*                                            normals,
        M const*                                            materials,
        NBinding                                            /* */
        ) -> decltype( simd::pack(
                surface<M>(),
                surface<M>(),
                surface<M>(),
                surface<M>()
                ) )
{
    auto hr4 = simd::unpack(hr);

    return simd::pack(
            surface<M>(
                hr4[0].hit ? get_normal(normals, hr4[0], NBinding()) : N(),
                hr4[0].hit ? materials[hr4[0].geom_id]               : M()
                ),
            surface<M>(
                hr4[1].hit ? get_normal(normals, hr4[1], NBinding()) : N(),
                hr4[1].hit ? materials[hr4[1].geom_id]               : M()
                ),
            surface<M>(
                hr4[2].hit ? get_normal(normals, hr4[2], NBinding()) : N(),
                hr4[2].hit ? materials[hr4[2].geom_id]               : M()
                ),
            surface<M>(
                hr4[3].hit ? get_normal(normals, hr4[3], NBinding()) : N(),
                hr4[3].hit ? materials[hr4[3].geom_id]               : M()
                )
            );
}


template <typename NBinding, typename N, typename TC, typename M, typename T>
inline auto get_surface_impl(
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        N const*                                            normals,
        TC const*                                           tex_coords,
        M const*                                            materials,
        T const*                                            textures,
        NBinding                                            /* */
        ) -> decltype( simd::pack(
                surface<M, vector<3, float>>(),
                surface<M, vector<3, float>>(),
                surface<M, vector<3, float>>(),
                surface<M, vector<3, float>>()
                ) )
{
    auto hr4 = simd::unpack(hr);

    auto tc4 = get_tex_coord(tex_coords, hr4);

    vector<3, float> cd4[4];
    for (unsigned i = 0; i < 4; ++i)
    {
        if (!hr4[i].hit)
        {
            continue;
        }

        auto const& tex = textures[hr4[i].geom_id];
        cd4[i] = tex.width() > 0 && tex.height() > 0 ? vector<3, float>(tex2D(tex, tc4[i])) : vector<3, float>(1.0);
    }

    return simd::pack(
            surface<M, vector<3, float>>(
                hr4[0].hit ? get_normal(normals, hr4[0], NBinding()) : N(),
                hr4[0].hit ? materials[hr4[0].geom_id]               : M(),
                hr4[0].hit ? cd4[0]                                  : vector<3, float>(1.0)
                ),
            surface<M, vector<3, float>>(
                hr4[1].hit ? get_normal(normals, hr4[1], NBinding()) : N(),
                hr4[1].hit ? materials[hr4[1].geom_id]               : M(),
                hr4[1].hit ? cd4[1]                                  : vector<3, float>(1.0)
                ),
            surface<M, vector<3, float>>(
                hr4[2].hit ? get_normal(normals, hr4[2], NBinding()) : N(),
                hr4[2].hit ? materials[hr4[2].geom_id]               : M(),
                hr4[2].hit ? cd4[2]                                  : vector<3, float>(1.0)
                ),
            surface<M, vector<3, float>>(
                hr4[3].hit ? get_normal(normals, hr4[3], NBinding()) : N(),
                hr4[3].hit ? materials[hr4[3].geom_id]               : M(),
                hr4[3].hit ? cd4[3]                                  : vector<3, float>(1.0)
                )
            );
}


//-------------------------------------------------------------------------------------------------
// Triangle / float4
//

template <typename NBinding, typename N, typename M>
inline auto get_surface_impl(
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        basic_triangle<3, float> const*                     primitives,
        N const*                                            normals,
        M const*                                            materials,
        NBinding                                            /* */
        ) -> decltype( get_surface_impl(hr, normals, materials, NBinding()) )
{
    VSNRAY_UNUSED(primitives);
    return get_surface_impl(hr, normals, materials, NBinding());
}


//-------------------------------------------------------------------------------------------------
// Generic primitive / float4
//

template <typename NBinding, unsigned C, typename N>
inline N simd_normal
(
    hit_record<simd::ray4, primitive<unsigned>> const& hr4,
    hit_record<ray, primitive<unsigned>> const& hr,
    generic_prim const* primitives,
    N const* normals,
    NBinding
)
{
    switch (hr.prim_type)
    {

    case detail::TrianglePrimitive:
        return get_normal(normals, hr, NBinding());

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

template <typename NBinding, typename N, typename M>
inline auto get_surface_impl(
        hit_record<simd::ray4, primitive<unsigned>> const&  hr,
        generic_prim const*                                 primitives,
        N const*                                            normals,
        M const*                                            materials,
        NBinding                                            /* */
        ) -> decltype( simd::pack(
                surface<M>(),
                surface<M>(),
                surface<M>(),
                surface<M>()
                ) )
{ 
    auto hr4 = simd::unpack(hr); 

    return simd::pack(
            surface<M>(
                hr4[0].hit ? simd_normal<0>(hr, hr4[0], primitives, normals, NBinding()) : N(),
                hr4[0].hit ? materials[hr4[0].geom_id]                                   : M()
                ),
            surface<M>(
                hr4[1].hit ? simd_normal<1>(hr, hr4[1], primitives, normals, NBinding()) : N(),
                hr4[1].hit ? materials[hr4[1].geom_id]                                   : M()
                ),
            surface<M>(
                hr4[2].hit ? simd_normal<2>(hr, hr4[2], primitives, normals, NBinding()) : N(),
                hr4[2].hit ? materials[hr4[2].geom_id]                                   : M()
                ),
            surface<M>(
                hr4[3].hit ? simd_normal<3>(hr, hr4[3], primitives, normals, NBinding()) : N(),
                hr4[3].hit ? materials[hr4[3].geom_id]                                   : M()
                )
            );
}


//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / float8
//

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename NBinding, typename N, typename M>
inline auto get_surface_impl(
        hit_record<simd::ray8, primitive<unsigned>> const&  hr,
        N const*                                            normals,
        M const*                                            materials,
        NBinding                                            /* */
        ) -> decltype( simd::pack(
                surface<M>(),
                surface<M>(),
                surface<M>(),
                surface<M>(),
                surface<M>(),
                surface<M>(),
                surface<M>(),
                surface<M>()
                ) )
{
    auto hr8 = simd::unpack(hr);

    return simd::pack(
            surface<M>(
                hr8[0].hit ? get_normal(normals, hr8[0], NBinding()) : N(),
                hr8[0].hit ? materials[hr8[0].geom_id]               : M()
                ),
            surface<M>(
                hr8[1].hit ? get_normal(normals, hr8[1], NBinding()) : N(),
                hr8[1].hit ? materials[hr8[1].geom_id]               : M()
                ),
            surface<M>(
                hr8[2].hit ? get_normal(normals, hr8[2], NBinding()) : N(),
                hr8[2].hit ? materials[hr8[2].geom_id]               : M()
                ),
            surface<M>(
                hr8[3].hit ? get_normal(normals, hr8[3], NBinding()) : N(),
                hr8[3].hit ? materials[hr8[3].geom_id]               : M()
                ),
            surface<M>(
                hr8[4].hit ? get_normal(normals, hr8[4], NBinding()) : N(),
                hr8[4].hit ? materials[hr8[4].geom_id]               : M()
                ),
            surface<M>(
                hr8[5].hit ? get_normal(normals, hr8[5], NBinding()) : N(),
                hr8[5].hit ? materials[hr8[5].geom_id]               : M()
                ),
            surface<M>(
                hr8[6].hit ? get_normal(normals, hr8[6], NBinding()) : N(),
                hr8[6].hit ? materials[hr8[6].geom_id]               : M()
                ),
            surface<M>(
                hr8[7].hit ? get_normal(normals, hr8[7], NBinding()) : N(),
                hr8[7].hit ? materials[hr8[7].geom_id]               : M()
                )
            );
}


//-------------------------------------------------------------------------------------------------
// Triangle / float8
//

template <typename NBinding, typename N, typename M>
inline auto get_surface_impl(
        hit_record<simd::ray8, primitive<unsigned>> const&  hr,
        basic_triangle<3, float> const*                     primitives,
        N const*                                            normals,
        M const*                                            materials,
        NBinding                                            /* */
        ) -> decltype( get_surface_impl(
                hr,
                normals,
                materials,
                NBinding()
                ) )
{
    VSNRAY_UNUSED(primitives);
    return get_surface_impl(hr, normals, materials, NBinding());
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

template <typename T>
VSNRAY_FUNC
inline auto has_emissive_material(surface<generic_mat<T>> const& surf)
    -> decltype( surf.material.get_type() == detail::EmissiveMaterial )
{
    return surf.material.get_type() == detail::EmissiveMaterial;
}

} // visionaray

#endif // VSNRAY_SURFACE_INL
