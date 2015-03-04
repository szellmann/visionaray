// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_MATH_INTERSECT_H
#define VSNRAY_MATH_INTERSECT_H

#include "simd/avx.h"
#include "simd/sse.h"

#include "aabb.h"
#include "plane.h"
#include "ray.h"
#include "sphere.h"
#include "triangle.h"
#include "vector.h"

namespace MATH_NAMESPACE
{

template <typename T1, typename T2>
struct hit_record;


//-------------------------------------------------------------------------------------------------
// ray / plane
//

template <typename T>
struct hit_record<basic_ray<T>, basic_plane<3, T>>
{

    typedef T value_type;

    MATH_FUNC hit_record() : hit(false) {}

    bool                    hit;
    value_type              t;
    vector<3, value_type>   pos;

};

template <typename T>
MATH_FUNC
inline hit_record<basic_ray<T>, basic_plane<3, T>> intersect
(
    basic_ray<T> const& ray, basic_plane<3, T> const& p
)
{

    hit_record<basic_ray<T>, basic_plane<3, T>> result;
    T s = dot(p.normal, ray.dir);

    if (s == T(0.0))
    {
        result.hit = false;
    }
    else
    {
        result.hit = true;
        result.t   = ( p.offset - dot(p.normal, ray.ori) ) / s;
        result.pos = ray.ori + result.t * ray.dir;
    }
    return result;

}


//-------------------------------------------------------------------------------------------------
// ray / aabb
//

template <typename T, typename U>
struct hit_record<basic_ray<T>, basic_aabb<U>>
{

    typedef T value_type;

    MATH_FUNC hit_record() : hit(false) {}

    bool            hit;
    value_type      tnear;
    value_type      tfar;

};

template <typename U>
struct hit_record<basic_ray<simd::float4>, basic_aabb<U>>
{

    simd::mask4     hit;
    simd::float4    tnear;
    simd::float4    tfar;

};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename U>
struct hit_record<basic_ray<simd::float8>, basic_aabb<U>>
{
    simd::mask8     hit;
    simd::float8    tnear;
    simd::float8    tfar;
};

#endif

template <typename T, typename U>
MATH_FUNC
inline hit_record<basic_ray<T>, basic_aabb<U>> intersect
(
    basic_ray<T> const& ray, basic_aabb<U> const& aabb, vector<3, T> inv_dir
)
{
    hit_record<basic_ray<T>, basic_aabb<U>> result;

    vector<3, T> t1 = (vector<3, T>(aabb.min) - ray.ori) * inv_dir;
    vector<3, T> t2 = (vector<3, T>(aabb.max) - ray.ori) * inv_dir;

    result.tnear = min_max( t1.x, t2.x, min_max(t1.y, t2.y, min(t1.z, t2.z)) );
    result.tfar  = max_min( t1.x, t2.x, max_min(t1.y, t2.y, max(t1.z, t2.z)) );
    result.hit   = result.tfar >= result.tnear && result.tfar >= T(0.0);

    return result;
}

template <typename T, typename U>
MATH_FUNC
inline hit_record<basic_ray<T>, basic_aabb<U>> intersect
(
    basic_ray<T> const& ray, basic_aabb<U> const& aabb
)
{
    vector<3, T> inv_dir = T(1.0) / ray.dir;
    return intersect(ray, aabb, inv_dir);
}


//-------------------------------------------------------------------------------------------------
// general ray primitive hit record
//

template <typename T>
struct hit_record<basic_ray<T>, primitive<unsigned>>
{

    typedef T value_type;

    bool hit;
    unsigned prim_type;
    unsigned prim_id;
    unsigned geom_id;

    value_type t;
    vector<3, value_type> isect_pos;

    value_type u;
    value_type v;

};

template <>
struct hit_record<simd::ray4, primitive<unsigned>>
{

    typedef simd::float4 value_type;

    simd::mask4 hit;
    simd::int4 prim_type;
    simd::int4 prim_id;
    simd::int4 geom_id;

    value_type t;
    vector<3, value_type> isect_pos;

    value_type u;
    value_type v;

};

VSNRAY_CPU_FUNC
inline std::array<hit_record<ray, primitive<unsigned>>, 4> unpack(hit_record<simd::ray4, primitive<unsigned>> const& hr)
{
    using namespace simd;

    VSNRAY_ALIGN(16) unsigned hit[4];
    store(hit, hr.hit.i);

    VSNRAY_ALIGN(16) unsigned prim_type[4];
    store(prim_type, hr.prim_type);

    VSNRAY_ALIGN(16) unsigned prim_id[4];
    store(prim_id, hr.prim_id);

    VSNRAY_ALIGN(16) unsigned geom_id[4];
    store(geom_id, hr.geom_id);

    VSNRAY_ALIGN(16) float t[4];
    store(t, hr.t);

    auto isect_pos = unpack(hr.isect_pos);

    VSNRAY_ALIGN(16) float u[4];
    store(u, hr.u);

    VSNRAY_ALIGN(16) float v[4];
    store(v, hr.v);

    return std::array<hit_record<ray, primitive<unsigned>>, 4>
    {{
        { hit[0] != 0, prim_type[0], prim_id[0], geom_id[0], t[0], isect_pos[0], u[0], v[0] },
        { hit[1] != 0, prim_type[1], prim_id[1], geom_id[1], t[1], isect_pos[1], u[1], v[1] },
        { hit[2] != 0, prim_type[2], prim_id[2], geom_id[2], t[2], isect_pos[2], u[2], v[2] },
        { hit[3] != 0, prim_type[3], prim_id[3], geom_id[3], t[3], isect_pos[3], u[3], v[3] },
    }};
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <>
struct hit_record<simd::ray8, primitive<unsigned>>
{

    typedef simd::float8 value_type;

    simd::mask8 hit;
    simd::int8 prim_type;
    simd::int8 prim_id;
    simd::int8 geom_id;

    value_type t;
    vector<3, value_type> isect_pos;

    value_type u;
    value_type v;

};

VSNRAY_CPU_FUNC
inline std::array<hit_record<ray, primitive<unsigned>>, 8> unpack(hit_record<simd::ray8, primitive<unsigned>> const& hr)
{
    using namespace simd;

    VSNRAY_ALIGN(32) unsigned hit[8];
    store(hit, hr.hit.i);

    VSNRAY_ALIGN(32) unsigned prim_type[8];
    store(prim_type, hr.prim_type);

    VSNRAY_ALIGN(32) unsigned prim_id[8];
    store(prim_id, hr.prim_id);

    VSNRAY_ALIGN(32) unsigned geom_id[8];
    store(geom_id, hr.geom_id);

    VSNRAY_ALIGN(32) float t[8];
    store(t, hr.t);

    auto isect_pos = unpack(hr.isect_pos);

    VSNRAY_ALIGN(32) float u[8];
    store(u, hr.u);

    VSNRAY_ALIGN(32) float v[8];
    store(v, hr.v);

    return std::array<hit_record<ray, primitive<unsigned>>, 8>
    {{
        { hit[0] != 0, prim_type[0], prim_id[0], geom_id[0], t[0], isect_pos[0], u[0], v[0] },
        { hit[1] != 0, prim_type[1], prim_id[1], geom_id[1], t[1], isect_pos[1], u[1], v[1] },
        { hit[2] != 0, prim_type[2], prim_id[2], geom_id[2], t[2], isect_pos[2], u[2], v[2] },
        { hit[3] != 0, prim_type[3], prim_id[3], geom_id[3], t[3], isect_pos[3], u[3], v[3] },
        { hit[4] != 0, prim_type[4], prim_id[4], geom_id[4], t[4], isect_pos[4], u[4], v[4] },
        { hit[5] != 0, prim_type[5], prim_id[5], geom_id[5], t[5], isect_pos[5], u[5], v[5] },
        { hit[6] != 0, prim_type[6], prim_id[6], geom_id[6], t[6], isect_pos[6], u[6], v[6] },
        { hit[7] != 0, prim_type[7], prim_id[7], geom_id[7], t[7], isect_pos[7], u[7], v[7] },
    }};
}

#endif


//-------------------------------------------------------------------------------------------------
// ray triangle
//

template <typename T, typename U>
MATH_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect
(
    basic_ray<T> const& ray, basic_triangle<3, U, unsigned> const& tri
)
{

    typedef vector<3, T> vec_type;

    hit_record<basic_ray<T>, primitive<unsigned>> result;
    result.t = T(-1.0);

    // case T != U
    vec_type v1(tri.v1);
    vec_type e1(tri.e1);
    vec_type e2(tri.e2);

    vec_type s1 = cross(ray.dir, e2);
    T div = dot(s1, e1);

    result.hit = ( div != T(0.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    T inv_div = T(1.0) / div;

    vec_type d = ray.ori - v1;
    T b1 = dot(d, s1) * inv_div;

    result.hit &= ( b1 >= T(0.0) && b1 <= T(1.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    vec_type s2 = cross(d, e1);
    T b2 = dot(ray.dir, s2) * inv_div;

    result.hit &= ( b2 >= T(0.0) && b1 + b2 <= T(1.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    result.prim_type = detail::TrianglePrimitive;
    result.prim_id = tri.prim_id;
    result.geom_id = tri.geom_id;
    result.t = dot(e2, s2) * inv_div;
    result.u = b1;
    result.v = b2;
    return result;

}


//-------------------------------------------------------------------------------------------------
// ray sphere
//

template <typename T, typename U>
MATH_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect
(
    basic_ray<T> const& ray, basic_sphere<U, unsigned> const& sphere
)
{

    typedef basic_ray<T> ray_type;
    typedef vector<3, T> vec_type;

    ray_type r = ray;
    r.ori -= vec_type( sphere.center );

    auto A = dot(r.dir, r.dir);
    auto B = dot(r.dir, r.ori) * T(2.0);
    auto C = dot(r.ori, r.ori) - sphere.radius * sphere.radius;

    // solve Ax**2 + Bx + C
    auto disc = B * B - T(4.0) * A * C;
    auto valid = disc >= T(0.0);

    auto root_disc = select(valid, sqrt(disc), disc);

    auto q = select( B < T(0.0), T(-0.5) * (B - root_disc), T(-0.5) * (B + root_disc) );

    auto t1 = q / A;
    auto t2 = C / q;

    hit_record<basic_ray<T>, primitive<unsigned>> result;
    result.hit = valid;
    result.prim_type = detail::SpherePrimitive;
    result.prim_id = sphere.prim_id;
    result.geom_id = sphere.geom_id;
    result.t = select( valid, select( t1 > t2, t2, t1 ), T(-1.0) );
    return result;
}


} // MATH_NAMESPACE

#endif // VSNRAY_MATH_INTERSECT_H
