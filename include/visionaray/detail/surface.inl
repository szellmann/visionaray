// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SURFACE_INL
#define VSNRAY_SURFACE_INL

#include <stdexcept>
#include <type_traits>

#include "../generic_prim.h"
#include "../tags.h"

namespace visionaray
{

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
// Primitive with precalculated normals / float
//

template <typename NBinding, typename R, typename N, typename M>
VSNRAY_FUNC
inline surface<M> get_surface
(
    hit_record<R, primitive<unsigned>> const& hr,
    N const* normals,
    M const* materials,
    NBinding
)
{
    return surface<M>
    (
        get_normal(normals, hr, NBinding()),
        materials[hr.geom_id]
    );
}

template <typename NBinding, typename R, typename N, typename M>
VSNRAY_FUNC
inline surface<M> get_surface
(
    hit_record<R, primitive<unsigned>> const& hr,
    N const* normals,
    M const* materials,
    NBinding()
)
{
    return surface<M>
    (
        get_normal(normals, hr, NBinding()),
        materials[hr.geom_id]
    );
}

template <typename R, typename NBinding, typename N, typename TC, typename M, typename T>
VSNRAY_FUNC
inline surface<M, vector<3, float>> get_surface
(
    hit_record<R, primitive<unsigned>> const& hr,
    N const* normals,
    TC const* tex_coords,
    M const* materials,
    T const* textures,
    std::integral_constant<unsigned, 3>,
    NBinding
)
{
    typedef typename R::scalar_type scalar_type;

    auto tc1 = tex_coords[hr.prim_id * 3];
    auto tc2 = tex_coords[hr.prim_id * 3 + 1];
    auto tc3 = tex_coords[hr.prim_id * 3 + 2];

    auto s2  = tc3 * hr.v;
    auto s3  = tc2 * hr.u;
    auto s1  = tc1 * (scalar_type(1.0) - (hr.u + hr.v));
    auto tc  = s1 + s2 + s3;

    auto const& tex = textures[hr.geom_id];
    auto cd = tex.width() > 0 && tex.height() > 0 ? vector<3, float>(tex2D(tex, tc)) : vector<3, float>(255.0);
    cd /= scalar_type(255.0);

    auto normal = get_normal(normals, hr, NBinding());
    return surface<M, vector<3, float>>( normal, materials[hr.geom_id], cd );
}


//-------------------------------------------------------------------------------------------------
// Triangle / float
//

template <typename R, typename NBinding, typename N, typename M>
VSNRAY_FUNC
inline surface<M> get_surface
(
    hit_record<R, primitive<unsigned>> const& hr,
    basic_triangle<3, float> const* primitives,
    N const* normals,
    M const* materials,
    NBinding
)
{
    VSNRAY_UNUSED(primitives);
    return get_surface(hr, normals, materials, NBinding());
}


//-------------------------------------------------------------------------------------------------
// Generic primitive / float
//

template <typename R, typename NBinding, typename N, typename M>
VSNRAY_FUNC
inline surface<M> get_surface
(
    hit_record<R, primitive<unsigned>> const& hr,
    generic_prim const* primitives,
    N const* normals,
    M const* materials,
    NBinding
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

template <template <typename> class B, typename NBinding, typename N, template <typename> class M, typename R>
VSNRAY_FUNC
inline surface<M<typename R::scalar_type>> get_surface
(
    hit_record<R, primitive<unsigned>> const& hr,
    B<basic_triangle<3, float>> const* tree,
    N const* normals,
    M<float> const* materials,
    NBinding
)
{
    VSNRAY_UNUSED(tree);
    return get_surface(hr, normals, materials, NBinding());
}

template <template <typename> class B, typename NBinding, typename N, typename TC, template <typename> class M, typename T, typename R>
VSNRAY_FUNC
inline surface<M<typename R::scalar_type>, vector<3, typename R::scalar_type>> get_surface
(
    hit_record<R, primitive<unsigned>> const& hr,
    B<basic_triangle<3, float>> const* tree,
    N const* normals,
    TC const* tex_coords,
    M<float> const* materials,
    T const* textures,
    NBinding
)
{
    VSNRAY_UNUSED(tree);
    return get_surface(hr, normals, tex_coords, materials, textures, std::integral_constant<unsigned, 3>{}, NBinding());
}


//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / float4
//

template <typename NBinding, typename N, template <typename> class M>
inline surface<M<simd::float4>> get_surface
(
    hit_record<simd::ray4, primitive<unsigned>> const& hr,
    N const* normals,
    M<float> const* materials,
    NBinding
)
{
    auto hr4 = unpack(hr);

    N n[4] =
    {
        hr4[0].hit ? get_normal(normals, hr4[0], NBinding()) : N(),
        hr4[1].hit ? get_normal(normals, hr4[1], NBinding()) : N(),
        hr4[2].hit ? get_normal(normals, hr4[2], NBinding()) : N(),
        hr4[3].hit ? get_normal(normals, hr4[3], NBinding()) : N()
    };

    return surface<M<simd::float4>>
    (
        vector<3, simd::float4>
        (
            simd::float4( n[0].x, n[1].x, n[2].x, n[3].x ),
            simd::float4( n[0].y, n[1].y, n[2].y, n[3].y ),
            simd::float4( n[0].z, n[1].z, n[2].z, n[3].z )
        ),

        pack
        (
            hr4[0].hit ? materials[hr4[0].geom_id] : M<float>(),
            hr4[1].hit ? materials[hr4[1].geom_id] : M<float>(),
            hr4[2].hit ? materials[hr4[2].geom_id] : M<float>(),
            hr4[3].hit ? materials[hr4[3].geom_id] : M<float>()
        )
    );
}


template <typename NBinding, typename N, typename TC, template <typename> class M, typename T>
inline surface<M<simd::float4>, vector<3, simd::float4>> get_surface
(
    hit_record<simd::ray4, primitive<unsigned>> const& hr,
    N const* normals,
    TC const* tex_coords,
    M<float> const* materials,
    T const* textures,
    std::integral_constant<unsigned, 3>,
    NBinding
)
{
    auto hr4 = unpack(hr);

    N n[4] =
    {
        get_normal(normals, hr4[0], NBinding()),
        get_normal(normals, hr4[1], NBinding()),
        get_normal(normals, hr4[2], NBinding()),
        get_normal(normals, hr4[3], NBinding())
    };

    typedef simd::float4 scalar_type;

    VSNRAY_ALIGN(16) float x1[] = { tex_coords[hr4[0].prim_id * 3].x, tex_coords[hr4[1].prim_id * 3].x, tex_coords[hr4[2].prim_id * 3].x, tex_coords[hr4[3].prim_id * 3].x };
    VSNRAY_ALIGN(16) float x2[] = { tex_coords[hr4[0].prim_id * 3 + 1].x, tex_coords[hr4[1].prim_id * 3 + 1].x, tex_coords[hr4[2].prim_id * 3 + 1].x, tex_coords[hr4[3].prim_id * 3 + 1].x };
    VSNRAY_ALIGN(16) float x3[] = { tex_coords[hr4[0].prim_id * 3 + 2].x, tex_coords[hr4[1].prim_id * 3 + 2].x, tex_coords[hr4[2].prim_id * 3 + 2].x, tex_coords[hr4[3].prim_id * 3 + 2].x };

    VSNRAY_ALIGN(16) float y1[] = { tex_coords[hr4[0].prim_id * 3].y, tex_coords[hr4[1].prim_id * 3].y, tex_coords[hr4[2].prim_id * 3].y, tex_coords[hr4[3].prim_id * 3].y };
    VSNRAY_ALIGN(16) float y2[] = { tex_coords[hr4[0].prim_id * 3 + 1].y, tex_coords[hr4[1].prim_id * 3 + 1].y, tex_coords[hr4[2].prim_id * 3 + 1].y, tex_coords[hr4[3].prim_id * 3 + 1].y };
    VSNRAY_ALIGN(16) float y3[] = { tex_coords[hr4[0].prim_id * 3 + 2].y, tex_coords[hr4[1].prim_id * 3 + 2].y, tex_coords[hr4[2].prim_id * 3 + 2].y, tex_coords[hr4[3].prim_id * 3 + 2].y };

    vector<2, simd::float4> tc1(x1, y1);
    vector<2, simd::float4> tc2(x2, y2);
    vector<2, simd::float4> tc3(x3, y3);

    auto s2 = tc3 * hr.v;
    auto s3 = tc2 * hr.u;
    auto s1 = tc1 * (scalar_type(1.0) - (hr.u + hr.v));
    auto tc = s1 + s2 + s3;

    vector<3, simd::float4> cd;
    for (unsigned i = 0; i < 4; ++i)
    {
        if (!hr4[i].hit)
        {
            continue;
        }

        VSNRAY_ALIGN(16) int maski[4] = { 0, 0, 0, 0 };
        maski[i] = 0xFFFFFFFF;
        simd::int4 mi(maski);
        simd::mask4 m(mi);

        auto const& tex = textures[hr4[i].geom_id];
        auto clr = tex.width() > 0 && tex.height() > 0 ? tex2D(tex, tc) : vector<3, simd::float4>(255.0);
        clr /= scalar_type(255.0);

        cd = select(m, clr, cd);
    }

    return surface<M<simd::float4>, vector<3, simd::float4>>
    (
        vector<3, simd::float4>
        (
            simd::float4( n[0].x, n[1].x, n[2].x, n[3].x ),
            simd::float4( n[0].y, n[1].y, n[2].y, n[3].y ),
            simd::float4( n[0].z, n[1].z, n[2].z, n[3].z )
        ),

        pack
        (
            hr4[0].hit ? materials[hr4[0].geom_id] : M<float>(),
            hr4[1].hit ? materials[hr4[1].geom_id] : M<float>(),
            hr4[2].hit ? materials[hr4[2].geom_id] : M<float>(),
            hr4[3].hit ? materials[hr4[3].geom_id] : M<float>()
        ),

        cd
    );
}


//-------------------------------------------------------------------------------------------------
// Triangle / float4
//

template <typename NBinding, typename N, template <typename> class M>
inline surface<M<simd::float4>> get_surface
(
    hit_record<simd::ray4, primitive<unsigned>> const& hr,
    basic_triangle<3, float> const* primitives,
    N const* normals,
    M<float> const* materials,
    NBinding
)
{
    VSNRAY_UNUSED(primitives);
    return get_surface(hr, normals, materials, NBinding());
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

template <typename NBinding, typename N, template <typename> class M>
inline surface<M<simd::float4>> get_surface 
( 
    hit_record<simd::ray4, primitive<unsigned>> const& hr ,
    generic_prim const* primitives,
    N const* normals, 
    M<float> const* materials,
    NBinding
) 
{ 
    auto hr4 = unpack(hr); 

    N n[4] = 
    { 
        hr4[0].hit ? simd_normal<0>(hr, hr4[0], primitives, normals, NBinding()) : N(),
        hr4[1].hit ? simd_normal<1>(hr, hr4[1], primitives, normals, NBinding()) : N(),
        hr4[2].hit ? simd_normal<2>(hr, hr4[2], primitives, normals, NBinding()) : N(),
        hr4[3].hit ? simd_normal<3>(hr, hr4[3], primitives, normals, NBinding()) : N()
    }; 

    return surface<M<simd::float4>>
    ( 
        vector<3, simd::float4> 
        ( 
            simd::float4( n[0].x, n[1].x, n[2].x, n[3].x ), 
            simd::float4( n[0].y, n[1].y, n[2].y, n[3].y ), 
            simd::float4( n[0].z, n[1].z, n[2].z, n[3].z ) 
        ),

        pack
        (
            hr4[0].hit ? materials[hr4[0].geom_id] : M<float>(),
            hr4[1].hit ? materials[hr4[1].geom_id] : M<float>(),
            hr4[2].hit ? materials[hr4[2].geom_id] : M<float>(),
            hr4[3].hit ? materials[hr4[3].geom_id] : M<float>()
        )
    ); 
}


//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / float8
//

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename NBinding, typename N, template <typename> class M>
inline surface<M<simd::float8>> get_surface
(
    hit_record<simd::ray8, primitive<unsigned>> const& hr,
    N const* normals,
    M<float> const* materials,
    NBinding
)
{
    VSNRAY_ALIGN(32) int prim_ids[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    store(&prim_ids[0], hr.prim_id, hr.hit, simd::int8(0));

    VSNRAY_ALIGN(32) int geom_ids[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    store(&geom_ids[0], hr.geom_id, hr.hit, simd::int8(0));


    N n[8] =
    {
        get_normal(normals, prim_ids[0]),
        get_normal(normals, prim_ids[1]),
        get_normal(normals, prim_ids[2]),
        get_normal(normals, prim_ids[3]),
        get_normal(normals, prim_ids[4]),
        get_normal(normals, prim_ids[5]),
        get_normal(normals, prim_ids[6]),
        get_normal(normals, prim_ids[7])
    };

    return surface<M<simd::float8>>
    (
        vector<3, simd::float8>
        (
            simd::float8( n[0].x, n[1].x, n[2].x, n[3].x, n[4].x, n[5].x, n[6].x, n[7].x ),
            simd::float8( n[0].y, n[1].y, n[2].y, n[3].y, n[4].y, n[5].y, n[6].y, n[7].y ),
            simd::float8( n[0].z, n[1].z, n[2].z, n[3].z, n[4].z, n[5].z, n[6].z, n[7].z )
        ),

        pack
        (
            materials[geom_ids[0]], materials[geom_ids[1]],
            materials[geom_ids[2]], materials[geom_ids[3]],
            materials[geom_ids[4]], materials[geom_ids[5]],
            materials[geom_ids[6]], materials[geom_ids[7]]
        )
    );
}


//-------------------------------------------------------------------------------------------------
// Triangle / float8
//

template <typename NBinding, typename N, template <typename> class M>
inline surface<M<simd::float8>> get_surface
(
    hit_record<simd::ray8, primitive<unsigned>> const& hr,
    basic_triangle<3, float> const* primitives,
    N const* normals,
    M<float> const* materials,
    NBinding
)
{
    VSNRAY_UNUSED(primitives);
    return get_surface(hr, normals, materials, NBinding());
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


//-------------------------------------------------------------------------------------------------
// Functions to deduce appropriate surface via parameter inspection
//

template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface(HR const& hr, Params const& p, detail::has_no_textures_tag)
    -> decltype( get_surface(hr, p.prims.begin, p.normals, p.materials, typename Params::normal_binding{}) )
{
    return get_surface(hr, p.prims.begin, p.normals, p.materials, typename Params::normal_binding{});
}

template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface(HR const& hr, Params const& p, detail::has_textures_tag)
    -> decltype( get_surface(hr, p.prims.begin, p.normals, p.tex_coords, p.materials, p.textures, typename Params::normal_binding{}) )
{
    return get_surface(hr, p.prims.begin, p.normals, p.tex_coords, p.materials, p.textures, typename Params::normal_binding{});
}

template <typename HR, typename Params>
VSNRAY_FUNC
inline auto get_surface(HR const& hr, Params const& p)
    -> decltype( get_surface(hr, p, detail::has_textures<Params>{}) )
{
    return get_surface(hr, p, detail::has_textures<Params>{});
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
