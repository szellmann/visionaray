// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_INTERSECT_H
#define VSNRAY_MATH_INTERSECT_H 1

#include <type_traits>

#include "simd/avx.h"
#include "simd/sse.h"

#include "aabb.h"
#include "limits.h"
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
    using value_type = T;
    using mask_type = typename simd::mask_type<T>::type;

    MATH_FUNC hit_record()
        : hit(false)
        , t(numeric_limits<float>::max())
        , pos(T(0.0))
    {
    }

    mask_type               hit;
    value_type              t;
    vector<3, value_type>   pos;

};

template <typename T>
MATH_FUNC
inline hit_record<basic_ray<T>, basic_plane<3, T>> intersect(
        basic_ray<T> const&         ray,
        basic_plane<3, T> const&    p
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
    using value_type = T;
    using mask_type = typename simd::mask_type<T>::type;

    MATH_FUNC hit_record()
        : hit(false)
        , tnear(numeric_limits<float>::max())
        , tfar(-numeric_limits<float>::max())
    {
    }

    mask_type   hit;
    value_type  tnear;
    value_type  tfar;

};

template <typename T, typename U>
MATH_FUNC
inline hit_record<basic_ray<T>, basic_aabb<U>> intersect(
        basic_ray<T> const&     ray,
        basic_aabb<U> const&    aabb,
        vector<3, T>            inv_dir
        )
{
    hit_record<basic_ray<T>, basic_aabb<U>> result;

    vector<3, T> t1 = (vector<3, T>(aabb.min) - ray.ori) * inv_dir;
    vector<3, T> t2 = (vector<3, T>(aabb.max) - ray.ori) * inv_dir;

    result.tnear = min_max( t1.x, t2.x, min_max(t1.y, t2.y, min(t1.z, t2.z)) );
    result.tfar  = max_min( t1.x, t2.x, max_min(t1.y, t2.y, max(t1.z, t2.z)) );
    result.hit   = result.tfar >= result.tnear;

    return result;
}

template <typename T, typename U>
MATH_FUNC
inline hit_record<basic_ray<T>, basic_aabb<U>> intersect(
        basic_ray<T> const& ray,
        basic_aabb<U> const& aabb
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
    using value_type = T;
    using int_type = typename simd::int_type<T>::type;
    using mask_type = typename simd::mask_type<T>::type;

    MATH_FUNC hit_record()
        : hit(false)
        , prim_id(0)
        , geom_id(0)
        , t(numeric_limits<float>::max())
        , u(0.0)
        , v(0.0)
    {
    }


    mask_type hit;
    int_type prim_id;
    int_type geom_id;

    value_type t;
    vector<3, value_type> isect_pos;

    value_type u;
    value_type v;
};


//-------------------------------------------------------------------------------------------------
// ray triangle
//

template <typename T, typename U>
MATH_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect(
        basic_ray<T> const&                     ray,
        basic_triangle<3, U, unsigned> const&   tri
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
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect(
        basic_ray<T> const&                 ray,
        basic_sphere<U, unsigned> const&    sphere
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
    result.prim_id = sphere.prim_id;
    result.geom_id = sphere.geom_id;
    result.t = select( valid, select( t1 > t2, t2, t1 ), T(-1.0) );
    return result;
}


//-------------------------------------------------------------------------------------------------
// Utility functions that can be reimplemented for user-supplied hit records
// Used by the traversal functions (linear traversal, BVH traversal, etc.)
//


//-------------------------------------------------------------------------------------------------
// Check if hit record <query> is valid and closer than hit record <reference> in ray's
// parameter space
//

// default implementation for hit_record<ray, primitive<unsigned>>
template <typename HR>
MATH_FUNC
auto is_closer(HR const& query, HR const& reference)
    -> decltype(query.t < reference.t)
{
    using T = decltype(query.t);

    return query.hit && ( query.t >= T(0.0) && query.t < reference.t );
}

// query is box, reference is general primitive

template <typename T, typename U, typename HR>
MATH_FUNC
auto is_closer(hit_record<basic_ray<T>, basic_aabb<U>> const& query, HR const& reference)
    -> decltype(query.tnear < reference.t)
{
    return query.hit && query.tnear < reference.t && query.tfar >= U(0.0);
}


//-------------------------------------------------------------------------------------------------
// is_closer() overload with max_t
//

template <typename HR, typename T>
MATH_FUNC
auto is_closer(HR const& query, HR const& reference, T max_t)
    -> decltype(is_closer(query, reference))
{
    return is_closer(query, reference) && query.t < max_t;
}


// default implementation for hit_record<ray, primitive<unsigned>>

//-------------------------------------------------------------------------------------------------
// Update hit record <dst> with params from hit record <src> if <cond> is met
//

// default implementation for hit_record<ray, primitive<unsigned>>
template <typename HR, typename Cond>
MATH_FUNC
void update_if(HR& dst, HR const& src, Cond const& cond)
{
    dst.hit        |= cond;
    dst.t           = select( cond, src.t, dst.t );
    dst.prim_id     = select( cond, src.prim_id, dst.prim_id );
    dst.geom_id     = select( cond, src.geom_id, dst.geom_id );
    dst.u           = select( cond, src.u, dst.u );
    dst.v           = select( cond, src.v, dst.v );
}


//-------------------------------------------------------------------------------------------------
// pack / unpack functions for hit records
//

namespace simd
{

// general primitive --------------------------------------

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
inline std::array<hit_record<ray, primitive<unsigned>>, num_elements<FloatT>::value> unpack(
        hit_record<basic_ray<FloatT>, primitive<unsigned>> const& hr
        )
{
    using float_array = typename aligned_array<FloatT>::type;
    using int_array = typename aligned_array<typename int_type<FloatT>::type>::type;

    int_array hit;
    store(hit, hr.hit.i);

    int_array prim_id;
    store(prim_id, hr.prim_id);

    int_array geom_id;
    store(geom_id, hr.geom_id);

    float_array t;
    store(t, hr.t);

    auto isect_pos = unpack(hr.isect_pos);

    float_array u;
    store(u, hr.u);

    float_array v;
    store(v, hr.v);

    std::array<hit_record<ray, primitive<unsigned>>, num_elements<FloatT>::value> result;
    for (size_t i = 0; i < num_elements<FloatT>::value; ++i)
    {
        result[i].hit       = hit[i] != 0;
        result[i].prim_id   = prim_id[i];
        result[i].geom_id   = geom_id[i];
        result[i].isect_pos = isect_pos[i];
        result[i].u         = u[i];
        result[i].v         = v[i];
    }
    return result;
}

} // simd


} // MATH_NAMESPACE

#endif // VSNRAY_MATH_INTERSECT_H
