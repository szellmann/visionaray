// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SURFACE_INL
#define VSNRAY_SURFACE_INL

#include <type_traits>

#include "../bvh.h"
#include "../generic_prim.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / float
//

template <typename R, typename N, typename M>
VSNRAY_FUNC
inline surface<M> get_surface
(
    N const* normals,
    M const* materials,
    hit_record<R, primitive<unsigned>> const& hr
)
{
    return surface<M>
    (
        normals[hr.prim_id],
        materials[hr.geom_id]
    );
}

template <typename R, typename N, typename TC, typename M, typename T>
inline surface<M, vector<3, float>> get_surface
(
    N const* normals,
    TC const* tex_coords,
    M const* materials,
    T const* textures,
    hit_record<R, primitive<unsigned>> const& hr,
    std::integral_constant<unsigned, 3>
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

    auto tex = textures[hr.geom_id];
    auto cd = tex.width() > 0 && tex.height() > 0 ? vector<3, float>(tex2D(tex, tc)) : vector<3, float>(255.0);
    cd /= scalar_type(255.0);

    return surface<M, vector<3, float>>( normals[hr.prim_id], materials[hr.geom_id], cd );
}


//-------------------------------------------------------------------------------------------------
// Triangle / float
//

template <typename R, typename N, typename M>
VSNRAY_FUNC
inline surface<M> get_surface
(
    basic_triangle<3, float> const* primitives,
    N const* normals,
    M const* materials,
    hit_record<R, primitive<unsigned>> const& hr
)
{
    VSNRAY_UNUSED(primitives);
    return get_surface(normals, materials, hr);
}


//-------------------------------------------------------------------------------------------------
// Generic primitive / float
//

template <typename R, typename N, typename M>
VSNRAY_FUNC
inline surface<M> get_surface
(
    generic_prim const* primitives,
    N const* normals,
    M const* materials,
    hit_record<R, primitive<unsigned>> const& hr
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

template <template <typename> class B, typename N, template <typename> class M, typename R>
inline surface<M<typename R::scalar_type>> get_surface
(
    B<basic_triangle<3, float>> const* tree,
    N const* normals,
    M<float> const* materials,
    hit_record<R, primitive<unsigned>> const& hr
)
{
    VSNRAY_UNUSED(tree);
    return get_surface(normals, materials, hr);
}

template <template <typename> class B, typename N, typename TC, template <typename> class M, typename T, typename R>
inline surface<M<typename R::scalar_type>, vector<3, typename R::scalar_type>> get_surface
(
    B<basic_triangle<3, float>> const* tree,
    N const* normals,
    TC const* tex_coords,
    M<float> const* materials,
    T const* textures,
    hit_record<R, primitive<unsigned>> const& hr
)
{
    VSNRAY_UNUSED(tree);
    return get_surface(normals, tex_coords, materials, textures, hr, std::integral_constant<unsigned, 3>{});
}


//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / float4
//

template <typename N, template <typename> class M>
inline surface<M<simd::float4>> get_surface
(
    N const* normals,
    M<float> const* materials,
    hit_record<simd::ray4, primitive<unsigned>> const& hr
)
{
    VSNRAY_ALIGN(16) int prim_ids[4];
    store(&prim_ids[0], hr.prim_id, hr.hit, simd::int4(0));

    VSNRAY_ALIGN(16) int geom_ids[4];
    store(&geom_ids[0], hr.geom_id, hr.hit, simd::int4(0));

    N n[4] =
    {
        normals[prim_ids[0]],
        normals[prim_ids[1]],
        normals[prim_ids[2]],
        normals[prim_ids[3]]
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
            materials[geom_ids[0]], materials[geom_ids[1]],
            materials[geom_ids[2]], materials[geom_ids[3]]
        )
    );
}

template <typename N, typename TC, template <typename> class M, typename T>
inline surface<M<simd::float4>, vector<3, simd::float4>> get_surface
(
    N const* normals,
    TC const* tex_coords,
    M<float> const* materials,
    T const* textures,
    hit_record<simd::ray4, primitive<unsigned>> const& hr,
    std::integral_constant<unsigned, 3>
)
{
    VSNRAY_ALIGN(16) int prim_ids[4];
    store(&prim_ids[0], hr.prim_id, hr.hit, simd::int4(0));

    VSNRAY_ALIGN(16) int geom_ids[4];
    store(&geom_ids[0], hr.geom_id, hr.hit, simd::int4(0));

    N n[4] =
    {
        normals[prim_ids[0]],
        normals[prim_ids[1]],
        normals[prim_ids[2]],
        normals[prim_ids[3]]
    };

    typedef simd::float4 scalar_type;

    VSNRAY_ALIGN(16) float x1[] = { tex_coords[prim_ids[0] * 3].x, tex_coords[prim_ids[1] * 3].x, tex_coords[prim_ids[2] * 3].x, tex_coords[prim_ids[3] * 3].x };
    VSNRAY_ALIGN(16) float x2[] = { tex_coords[prim_ids[0] * 3 + 1].x, tex_coords[prim_ids[1] * 3 + 1].x, tex_coords[prim_ids[2] * 3 + 1].x, tex_coords[prim_ids[3] * 3 + 1].x };
    VSNRAY_ALIGN(16) float x3[] = { tex_coords[prim_ids[0] * 3 + 2].x, tex_coords[prim_ids[1] * 3 + 2].x, tex_coords[prim_ids[2] * 3 + 2].x, tex_coords[prim_ids[3] * 3 + 2].x };

    VSNRAY_ALIGN(16) float y1[] = { tex_coords[prim_ids[0] * 3].y, tex_coords[prim_ids[1] * 3].y, tex_coords[prim_ids[2] * 3].y, tex_coords[prim_ids[3] * 3].y };
    VSNRAY_ALIGN(16) float y2[] = { tex_coords[prim_ids[0] * 3 + 1].y, tex_coords[prim_ids[1] * 3 + 1].y, tex_coords[prim_ids[2] * 3 + 1].y, tex_coords[prim_ids[3] * 3 + 1].y };
    VSNRAY_ALIGN(16) float y3[] = { tex_coords[prim_ids[0] * 3 + 2].y, tex_coords[prim_ids[1] * 3 + 2].y, tex_coords[prim_ids[2] * 3 + 2].y, tex_coords[prim_ids[3] * 3 + 2].y };

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
        VSNRAY_ALIGN(16) int maski[4] = { 0, 0, 0, 0 };
        maski[i] = 0xFF;
        simd::mask4 m(maski);

        auto tex = textures[geom_ids[i]];
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
            materials[geom_ids[0]], materials[geom_ids[1]],
            materials[geom_ids[2]], materials[geom_ids[3]]
        ),

        cd
    );
}


//-------------------------------------------------------------------------------------------------
// Triangle / float4
//

template <typename N, template <typename> class M>
inline surface<M<simd::float4>> get_surface
(
    basic_triangle<3, float> const* primitives,
    N const* normals,
    M<float> const* materials,
    hit_record<simd::ray4, primitive<unsigned>> const& hr
)
{
    VSNRAY_UNUSED(primitives);
    return get_surface(normals, materials, hr);
}


//-------------------------------------------------------------------------------------------------
// Generic primitive / float4
//

template <unsigned C, typename N>
inline N simd_normal
(
    generic_prim const* primitives,
    N const* normals,
    hit_record<simd::ray4, primitive<unsigned>> const& hr,
    unsigned prim_type, unsigned prim_id
)
{

    switch (prim_type)
    {

    case detail::TrianglePrimitive:
        return normals[prim_id];

    case detail::SpherePrimitive:
    {
        basic_sphere<float> sphere = primitives[prim_id].sphere;

        auto tmp = hr.isect_pos;
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

template <typename N, template <typename> class M>
inline surface<M<simd::float4>> get_surface 
( 
    generic_prim const* primitives,
    N const* normals, 
    M<float> const* materials,
    hit_record<simd::ray4, primitive<unsigned>> const& hr 
) 
{ 
 
    VSNRAY_ALIGN(16) int types[4]; 
    store(&types[0], hr.prim_type, hr.hit, simd::int4(0)); 
 
    VSNRAY_ALIGN(16) int prim_ids[4]; 
    store(&prim_ids[0], hr.prim_id, hr.hit, simd::int4(0));

    VSNRAY_ALIGN(16) int geom_ids[4];
    store(&geom_ids[0], hr.geom_id, hr.hit, simd::int4(0));

    N n[4] = 
    { 
        simd_normal<0>(primitives, normals, hr, types[0], prim_ids[0]), 
        simd_normal<1>(primitives, normals, hr, types[1], prim_ids[1]), 
        simd_normal<2>(primitives, normals, hr, types[2], prim_ids[2]), 
        simd_normal<3>(primitives, normals, hr, types[3], prim_ids[3]) 
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
            materials[geom_ids[0]], materials[geom_ids[1]],
            materials[geom_ids[2]], materials[geom_ids[3]]
        )
    ); 
 
}


//-------------------------------------------------------------------------------------------------
// Primitive with precalculated normals / float8
//

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename N, template <typename> class M>
inline surface<M<simd::float8>> get_surface
(
    N const* normals,
    M<float> const* materials,
    hit_record<simd::ray8, primitive<unsigned>> const& hr
)
{
    VSNRAY_ALIGN(32) int prim_ids[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    store(&prim_ids[0], hr.prim_id, hr.hit, simd::int8(0));

    VSNRAY_ALIGN(32) int geom_ids[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    store(&geom_ids[0], hr.geom_id, hr.hit, simd::int8(0));


    N n[8] =
    {
        normals[prim_ids[0]],
        normals[prim_ids[1]],
        normals[prim_ids[2]],
        normals[prim_ids[3]],
        normals[prim_ids[4]],
        normals[prim_ids[5]],
        normals[prim_ids[6]],
        normals[prim_ids[7]]
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

template <typename N, template <typename> class M>
inline surface<M<simd::float8>> get_surface
(
    basic_triangle<3, float> const* primitives,
    N const* normals,
    M<float> const* materials,
    hit_record<simd::ray8, primitive<unsigned>> const& hr
)
{
    VSNRAY_UNUSED(primitives);
    return get_surface(normals, materials, hr);
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


//-------------------------------------------------------------------------------------------------
// Functions to deduce appropriate surface via parameter inspection
//

template <typename Params, typename HR>
inline auto get_surface(Params const& p, HR const& hr, detail::has_no_textures_tag)
    -> decltype( get_surface(p.prims.begin, p.normals, p.materials, hr) )
{
    return get_surface(p.prims.begin, p.normals, p.materials, hr);
}

template <typename Params, typename HR>
inline auto get_surface(Params const& p, HR const& hr, detail::has_textures_tag)
    -> decltype( get_surface(p.prims.begin, p.normals, p.tex_coords, p.materials, p.textures, hr) )
{
    return get_surface(p.prims.begin, p.normals, p.tex_coords, p.materials, p.textures, hr);
}

template <typename Params, typename HR>
inline auto get_surface(Params const& p, HR const& hr)
    -> decltype( get_surface(p, hr, detail::has_textures<Params>{}) )
{
    return get_surface(p, hr, detail::has_textures<Params>{});
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
inline bool has_emissive_material(surface<emissive<T>> const& surf)
{
    VSNRAY_UNUSED(surf);
    return true;
}

template <typename T>
VSNRAY_FUNC
inline bool has_emissive_material(surface<generic_mat<T>> const& surf)
{
    return surf.material.get_type() == detail::EmissiveMaterial;
}

} // visionaray

#endif // VSNRAY_SURFACE_INL


