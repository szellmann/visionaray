// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_INTERSECT_H
#define VSNRAY_MATH_INTERSECT_H 1

#include <type_traits>

#include "simd/type_traits.h"
#include "aabb.h"
#include "config.h"
#include "cylinder.h"
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
// ray / aabb
//

template <typename T, typename U>
struct hit_record<basic_ray<T>, basic_aabb<U>>
{
    using scalar_type = T;
    using mask_type = simd::mask_type_t<T>;

    mask_type   hit   = mask_type(false);
    scalar_type tnear =  numeric_limits<T>::max();
    scalar_type tfar  = -numeric_limits<T>::max();
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

//-------------------------------------------------------------------------------------------------
// simd overload: ray1 / aabbN
//

// TODO: combine those three using type traits!
template <>
struct hit_record<basic_ray<float>, basic_aabb<simd::float4>>
{
    using T = simd::float4;

    using scalar_type = T;
    using mask_type = simd::mask_type_t<T>;

    mask_type   hit   = mask_type(false);
    scalar_type tnear =  numeric_limits<T>::max();
    scalar_type tfar  = -numeric_limits<T>::max();
};

template <>
struct hit_record<basic_ray<float>, basic_aabb<simd::float8>>
{
    using T = simd::float8;

    using scalar_type = T;
    using mask_type = simd::mask_type_t<T>;

    mask_type   hit   = mask_type(false);
    scalar_type tnear =  numeric_limits<T>::max();
    scalar_type tfar  = -numeric_limits<T>::max();
};

template <>
struct hit_record<basic_ray<float>, basic_aabb<simd::float16>>
{
    using T = simd::float16;

    using scalar_type = T;
    using mask_type = simd::mask_type_t<T>;

    mask_type   hit   = mask_type(false);
    scalar_type tnear =  numeric_limits<T>::max();
    scalar_type tfar  = -numeric_limits<T>::max();
};

template <typename T, typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type>
MATH_FUNC
inline hit_record<basic_ray<float>, basic_aabb<T>> intersect(
        basic_ray<float> const& ray,
        basic_aabb<T> const&    aabb,
        vector<3, float>        inv_dir
        )
{
    hit_record<basic_ray<float>, basic_aabb<T>> result;

    vector<3, T> t1 = (aabb.min - vector<3, T>(ray.ori)) * vector<3, T>(inv_dir);
    vector<3, T> t2 = (aabb.max - vector<3, T>(ray.ori)) * vector<3, T>(inv_dir);

    vector<3, T> tmin = min(t1, t2);
    vector<3, T> tmax = max(t1, t2);

    result.tnear = max(ray.tmin, max(tmin.x, max(tmin.y, tmin.z)));
    result.tfar  = min(ray.tmax, min(tmax.x, min(tmax.y, tmax.z)));
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
    vector<3, T> inv_dir(
        select(ray.dir.x != T(0.0), T(1.0) / ray.dir.x, T(FLT_MAX)),
        select(ray.dir.y != T(0.0), T(1.0) / ray.dir.y, T(FLT_MAX)),
        select(ray.dir.z != T(0.0), T(1.0) / ray.dir.z, T(FLT_MAX))
        );
    return intersect(ray, aabb, inv_dir);
}


//-------------------------------------------------------------------------------------------------
// general ray primitive hit record
//

template <typename R>
struct hit_record<R, primitive<unsigned>>
{
    using T = typename R::scalar_type;
    using scalar_type = T;
    using int_type = simd::int_type_t<T>;
    using mask_type = simd::mask_type_t<T>;

    mask_type hit          = mask_type(false);
    int_type prim_id       = int_type(0);
    int_type geom_id       = int_type(0);
    int_type inst_id       = int_type(-1);

    T t                    = numeric_limits<T>::max();
    vector<3, T> isect_pos;

    T u                    = T(0.0);
    T v                    = T(0.0);
};


//-------------------------------------------------------------------------------------------------
// ray / triangle
//

template <typename R, typename U>
MATH_FUNC
inline hit_record<R, primitive<unsigned>> intersect(R const& ray, basic_triangle<3, U, unsigned> const& tri)
{
#if 1
    using T = typename R::scalar_type;
    using vec_type = vector<3, T>;

    hit_record<R, primitive<unsigned>> result;
    result.t = T(-1.0);

    // case T != U
    const vec_type v1(tri.v1);
    const vec_type e1(tri.e1);
    const vec_type e2(tri.e2);

    const vec_type ori(ray.ori);
    const vec_type dir(ray.dir);

    const vec_type N = cross(e1, e2);
    result.hit = (N != vec_type(T(0.0)));
    if (!any(result.hit))
    {
        return result;
    }

    result.hit &= abs(dot(dir, N)) >= T(1e-12);
    if (!any(result.hit))
    {
        return result;
    }

    const T t = -dot(ori - v1, N) / dot(dir, N);
    result.hit &= (t >= T(ray.tmin) && t <= T(ray.tmax));
    if (!any(result.hit))
    {
        return result;
    }

    const vec_type P = ori - v1 + t * dir;

    const auto in_x = abs(N.x) >= max(abs(N.y), abs(N.z));
    const auto in_y = !in_x && (abs(N.y) > abs(N.z));
    const auto in_z = !in_x && !in_y;

    const T e1u = select(in_x, e1.y, e1.x);
    const T e2u = select(in_x, e2.y, e2.x);
    const T Pu  = select(in_x, P.y, P.x);

    const T e1v = select(in_z, e1.y, e1.z);
    const T e2v = select(in_z, e2.y, e2.z);
    const T Pv  = select(in_z, P.y, P.z);

    const T det = det2(e1u, e1v, e2u, e2v);
    result.hit &= (det != T(0.0));
    if (!any(result.hit))
    {
        return result;
    }

    const T u = det2(Pu, e2u, Pv, e2v) / det2(e1u, e2u, e1v, e2v);
    const T v = det2(e1u, Pu, e1v, Pv) / det2(e1u, e2u, e1v, e2v);
    result.hit &= (u >= T(0.0) && v >= T(0.0) && (u + v) <= T(1.0));
    if (!any(result.hit))
    {
        return result;
    }

    result.prim_id = tri.prim_id;
    result.geom_id = tri.geom_id;
    result.t = t;
    result.u = u;
    result.v = v;
    return result;
#else
    using T = typename R::scalar_type;
    using vec_type = vector<3, T>;

    hit_record<R, primitive<unsigned>> result;
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
#endif
}

//-------------------------------------------------------------------------------------------------
// simd overload: ray1 / triangleN
//

template <typename R>
struct hit_record<R, primitive<simd::int4>>
{
    using T = simd::float4;
    using scalar_type = T;
    using int_type = simd::int_type_t<T>;
    using mask_type = simd::mask_type_t<T>;

    mask_type hit          = mask_type(false);
    int_type prim_id       = int_type(0);
    int_type geom_id       = int_type(0);

    T t                    = numeric_limits<T>::max();
    vector<3, T> isect_pos;

    T u                    = T(0.0);
    T v                    = T(0.0);
};

template <
    typename T,
    typename I = typename simd::int_type<T>::type,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
MATH_FUNC
inline hit_record<basic_ray<float>, primitive<I>> intersect(
        basic_ray<float> const& ray,
        basic_triangle<3, T, I> const& tri
        )
{
    using vec_type = vector<3, T>;

    hit_record<basic_ray<float>, primitive<I>> result;
    result.t = T(-1.0);

    vec_type ori(ray.ori);
    vec_type dir(ray.dir);

    vec_type s1 = cross(dir, tri.e2);
    T div = dot(s1, tri.e1);

    result.hit = ( div != T(0.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    T inv_div = T(1.0) / div;

    vec_type d = ori - tri.v1;
    T b1 = dot(d, s1) * inv_div;

    result.hit &= ( b1 >= T(0.0) && b1 <= T(1.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    vec_type s2 = cross(d, tri.e1);
    T b2 = dot(dir, s2) * inv_div;

    result.hit &= ( b2 >= T(0.0) && b1 + b2 <= T(1.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    result.prim_id = tri.prim_id;
    result.geom_id = tri.geom_id;
    result.t = dot(tri.e2, s2) * inv_div;
    result.u = b1;
    result.v = b2;
    return result;
}


//-------------------------------------------------------------------------------------------------
// ray / sphere
//

template <typename R, typename U>
MATH_FUNC
inline hit_record<R, primitive<unsigned>> intersect(R const& ray, basic_sphere<U, unsigned> const& sphere)
{
    using T = typename R::scalar_type;
    using vec_type = vector<3, T>;

    R r = ray;
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

    hit_record<R, primitive<unsigned>> result;
    result.hit = valid && (t1 >= T(0.0) || t2 >= T(0.0));
    result.prim_id = sphere.prim_id;
    result.geom_id = sphere.geom_id;
    result.t = T(-1.0);
    result.t = select(t1 >= T(0.0) && t2 >= T(0.0), min(t1, t2), result.t);
    result.t = select(t1 >= T(0.0) && t2 <  T(0.0), t1,          result.t);
    result.t = select(t1 <  T(0.0) && t2 >= T(0.0), t2,          result.t);
    return result;
}


//-------------------------------------------------------------------------------------------------
// ray / cylinder
//

template <typename R, typename U>
MATH_FUNC
inline hit_record<R, primitive<unsigned>> intersect(R const& ray, basic_cylinder<U, unsigned> const& cylinder)
{
    using T = typename R::scalar_type;
    using vec_type = vector<3, T>;

    vec_type ori = ray.ori;
    T min_dist = max(
        T(0.0),
        min(
            length(cylinder.v1 - ori) - cylinder.radius * T(2.0),
            length(cylinder.v2 - ori) - cylinder.radius * T(2.0))
        );

    ori = ori + min_dist * ray.dir;

    vec_type ba = cylinder.v2 - cylinder.v1;
    vec_type oc = ori - cylinder.v1;

    T baba = dot(ba, ba);
    T bard = dot(ba, ray.dir);
    T baoc = dot(ba, oc);

    T k2 = baba - bard * bard;
    T k1 = baba * dot(oc, ray.dir) - baoc * bard;
    T k0 = baba * dot(oc, oc) - baoc * baoc - cylinder.radius * cylinder.radius * baba;

    T h = k1 * k1 - k2 * k0;

    auto valid = h >= T(0.0);

    h = select(valid, sqrt(h), h);
    T t = (-k1 - h) / k2;

    hit_record<R, primitive<unsigned>> result;
    result.t = T(-1.0);

    // body
    T y = baoc + t * bard;

    auto hit1 = valid && y > T(0.0) && y < baba;
    result.t = select(hit1, min_dist + t, result.t);

    // caps
    t = select(y < T(0.0), -baoc / bard, (baba - baoc) / bard);
    auto hit2 = valid && abs(k1 + k2 * t) < h;

    result.t = select(!hit1 && hit2, min_dist + t, result.t);
    result.u = clamp(y / baba, T(0.0), T(1.0));
    result.hit = hit1 || hit2;
    result.prim_id = cylinder.prim_id;
    result.geom_id = cylinder.geom_id;
    return result;
}


//-------------------------------------------------------------------------------------------------
// ray / plane
//

template <typename R, typename U>
MATH_FUNC
inline hit_record<R, primitive<unsigned>> intersect(R const& ray, basic_plane<3, U, unsigned> const& p)
{
    using T = typename R::scalar_type;

    hit_record<R, primitive<unsigned>> result;

    vector<3, T> normal(p.normal);
    T offset(p.offset);

    T s = dot(normal, ray.dir);

    result.hit = (s != T(0.0));

    result.prim_id = p.prim_id;
    result.geom_id = p.geom_id;

    result.t = select(
            result.hit,
            ( offset - dot(normal, ray.ori) ) / s,
            T(-1.0)
            );

    result.isect_pos = select(
            result.hit,
            ray.ori + result.t * ray.dir,
            result.isect_pos
            ); // TODO: sphere/triangle don't assign isect_pos in intersect

    return result;
}


//-------------------------------------------------------------------------------------------------
// plane / sphere
//

template <typename T, typename P>
struct hit_record<basic_plane<3, T>, basic_sphere<T, P>>
{
    using scalar_type = T;
    using mask_type = simd::mask_type_t<T>;

    // TODO: make this part of the API?
    struct basic_circle
    {
        vector<3, T> center;
        T radius;
    };

    MATH_FUNC hit_record()
        : hit(false)
    {
    }

    mask_type       hit;
    basic_circle    circle;
};

template <typename T, typename P>
MATH_FUNC
inline hit_record<basic_plane<3, T>, basic_sphere<T, P>> intersect(
        basic_plane<3, T> const&    plane,
        basic_sphere<T, P> const&   sphere
        )
{
    hit_record<basic_plane<3, T>, basic_sphere<T, P>> result;

    T dist = plane.offset - dot(sphere.center, plane.normal);

    auto hit = abs(dist) <= sphere.radius;

    result.hit           = hit;
    result.circle.center = select(hit, sphere.center + plane.normal * dist, vector<3, T>());
    result.circle.radius = select(hit, sqrt(sphere.radius * sphere.radius - dist * dist), T(0.0));

    return result;
}

template <typename T, typename P>
MATH_FUNC
inline hit_record<basic_sphere<T, P>, basic_plane<3, T>> intersect(
        basic_sphere<T, P> const&   sphere,
        basic_plane<3, T> const&    plane
        )
{
    return intersect(plane, sphere);
}


//-------------------------------------------------------------------------------------------------
// pack / unpack functions for hit records
//

namespace simd
{

// general primitive --------------------------------------

template <
    unsigned N,
    typename T = float_from_simd_width_t<N>
    >
MATH_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> pack(
        array<hit_record<ray, primitive<unsigned>>, N> const& hrs
        )
{
    hit_record<basic_ray<T>, primitive<unsigned>> result;

    int* hit = reinterpret_cast<int*>(&result.hit);
    int* prim_id = reinterpret_cast<int*>(&result.prim_id);
    int* geom_id = reinterpret_cast<int*>(&result.geom_id);
    float* t = reinterpret_cast<float*>(&result.t);
    array<vec3, N> isect_pos;
    float* u = reinterpret_cast<float*>(&result.u);
    float* v = reinterpret_cast<float*>(&result.v);

    for (unsigned i = 0; i < N; ++i)
    {
        hit[i]       = hrs[i].hit ? 0xFFFFFFFF : 0x00000000;
        prim_id[i]   = hrs[i].prim_id;
        geom_id[i]   = hrs[i].geom_id;
        t[i]         = hrs[i].t;
        isect_pos[i] = hrs[i].isect_pos;
        u[i]         = hrs[i].u;
        v[i]         = hrs[i].v;
    }

    result.isect_pos = pack(isect_pos);

    return result;
}

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
MATH_FUNC
inline array<hit_record<ray, primitive<unsigned>>, num_elements<FloatT>::value> unpack(
        hit_record<basic_ray<FloatT>, primitive<unsigned>> const& hr
        )
{
    using float_array = aligned_array_t<FloatT>;
    using int_array = aligned_array_t<int_type_t<FloatT>>;

    int_array hit;
    store(hit, convert_to_int(hr.hit));

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

    array<hit_record<ray, primitive<unsigned>>, num_elements<FloatT>::value> result;
    for (unsigned i = 0; i < num_elements<FloatT>::value; ++i)
    {
        result[i].hit       = hit[i] != 0;
        result[i].prim_id   = prim_id[i];
        result[i].geom_id   = geom_id[i];
        result[i].t         = t[i];
        result[i].isect_pos = isect_pos[i];
        result[i].u         = u[i];
        result[i].v         = v[i];
    }
    return result;
}

// TODO: consolidate!
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
template <>
inline array<hit_record<ray, primitive<unsigned>>, num_elements<simd::float16>::value> unpack(
        hit_record<basic_ray<simd::float16>, primitive<unsigned>> const& hr
        )
{
    using FloatT = simd::float16;
    using float_array = aligned_array_t<FloatT>;
    using int_array = aligned_array_t<int_type_t<FloatT>>;
    using mask_array = aligned_array_t<mask_type_t<FloatT>>;

    mask_array hit;
    store(hit, hr.hit);

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

    array<hit_record<ray, primitive<unsigned>>, num_elements<FloatT>::value> result;
    for (unsigned i = 0; i < num_elements<FloatT>::value; ++i)
    {
        result[i].hit       = hit[i] != 0;
        result[i].prim_id   = prim_id[i];
        result[i].geom_id   = geom_id[i];
        result[i].t         = t[i];
        result[i].isect_pos = isect_pos[i];
        result[i].u         = u[i];
        result[i].v         = v[i];
    }
    return result;
}
#endif

} // simd

} // MATH_NAMESPACE

#endif // VSNRAY_MATH_INTERSECT_H
