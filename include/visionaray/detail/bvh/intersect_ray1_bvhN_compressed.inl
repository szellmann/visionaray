// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <iostream>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <visionaray/math/simd/type_traits.h>
#include <visionaray/math/matrix.h>
#include <visionaray/intersector.h>
#include <visionaray/update_if.h>

#include "../tags.h"
#include "hit_record.h"

// #define likely(x)   __builtin_expect(!!(x), 1)
// #define unlikely(x) __builtin_expect(!!(x), 0)

namespace visionaray
{

template <
    detail::traversal_type Traversal,
    typename R,
    typename BVH,
    typename Intersector,
    typename T = typename R::scalar_type
    >
VSNRAY_FUNC
inline auto intersect_ray1_bvhN_compressed(
        R const&     ray,
        BVH const&   b,
        Intersector& isect
        )
    -> hit_record_bvh<
            R,
            decltype( isect(ray, std::declval<typename BVH::primitive_type>()) )
            >
{
    using namespace detail;
    using HR = hit_record_bvh<R, decltype(isect(ray, std::declval<typename BVH::primitive_type>()))>;

    HR result;

    using N = bvh_compressed_node<BVH::Width>;

    struct stack_entry
    {
        typename N::Child addr;
        unsigned dist;
    };

    stack_entry stack[64];
    char ptr = 0;
    stack[ptr++] = { { 0, 0 }, 0 }; // root node

    vector<3, T> inv_dir(
        select(ray.dir.x != T(0.0), T(1.0) / ray.dir.x, T(FLT_MAX)),
        select(ray.dir.y != T(0.0), T(1.0) / ray.dir.y, T(FLT_MAX)),
        select(ray.dir.z != T(0.0), T(1.0) / ray.dir.z, T(FLT_MAX))
        );

    // while ray not terminated
next:
    while (ptr > 0)
    {
        auto se = stack[--ptr];
        typename N::Child addr = se.addr;
        unsigned dist = se.dist;

        // while node does not contain primitives
        //     traverse to the next node

        while (addr.num_prims == 0)
        {
            if (*((unsigned*)&result.t) < dist)
            {
                goto next;
            }

            const auto &node = b.node(addr.id);

            using F = simd::float_from_simd_width_t<BVH::Width>;
            using I = simd::int_from_simd_width_t<BVH::Width>;

            I minx_ext, miny_ext, minz_ext, maxx_ext, maxy_ext, maxz_ext;
            simd::sign_extend(minx_ext, node.child_bounds.minx);
            simd::sign_extend(miny_ext, node.child_bounds.miny);
            simd::sign_extend(minz_ext, node.child_bounds.minz);
            simd::sign_extend(maxx_ext, node.child_bounds.maxx);
            simd::sign_extend(maxy_ext, node.child_bounds.maxy);
            simd::sign_extend(maxz_ext, node.child_bounds.maxz);

            basic_aabb<F> aabbN;

            aabbN.min.x = convert_to_float(minx_ext);
            aabbN.min.y = convert_to_float(miny_ext);
            aabbN.min.z = convert_to_float(minz_ext);

            aabbN.max.x = convert_to_float(maxx_ext);
            aabbN.max.y = convert_to_float(maxy_ext);
            aabbN.max.z = convert_to_float(maxz_ext);

            auto pow2 = [](char e) {
                unsigned u((e + 127) << 23);
                return *(float*)&u;
            };

            // This uses the optimization from the Ylitie paper
            // transforming the ray into the coordinate system of
            // the local grid:
            vec3 P(pow2(node.e[0]), pow2(node.e[1]), pow2(node.e[2]));
            vector<3, F> local_ori((node.origin - ray.ori) * inv_dir);
            vector<3, F> local_dir(P * inv_dir);

            hit_record<basic_ray<float>, basic_aabb<F>> hrN;
            vector<3, F> t1 = aabbN.min * local_dir + local_ori;
            vector<3, F> t2 = aabbN.max * local_dir + local_ori;

            vector<3, F> tmin = min(t1, t2);
            vector<3, F> tmax = max(t1, t2);

            hrN.tnear = max(ray.tmin, max(tmin.x, max(tmin.y, tmin.z)));
            hrN.tfar  = min(ray.tmax, min(tmax.x, min(tmax.y, tmax.z)));
            hrN.hit   = hrN.tfar >= hrN.tnear;

            hrN.hit &= aabbN.min.x <= aabbN.max.x;
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_NEON_FP)
            hrN.hit &= reinterpret_as_uint(hrN.tnear) < reinterpret_as_uint(F(result.t));
            hrN.hit &= reinterpret_as_uint(hrN.tfar) >= reinterpret_as_uint(F(ray.tmin)) &&
                       reinterpret_as_uint(hrN.tnear) <= reinterpret_as_uint(F(ray.tmax));
#else
            // TODO: unsigned comparisons with SSE, and check if this is worth it..
            hrN.hit &= hrN.tnear < F(result.t);
            hrN.hit &= hrN.tfar >= F(ray.tmin) &&
                       hrN.tnear <= F(ray.tmax);
#endif

            auto mask = movemask(hrN.hit.i);

            if (!mask)
            {
                goto next;
            }

            unsigned* tnear = reinterpret_cast<unsigned*>(&hrN.tnear);

            auto bsf = [](int& m) {
                int i =  ctz(m);
                m &= m-1;
                return i;
            };

            if constexpr (Traversal == detail::ClosestHit)
            {
                int i1 = bsf(mask);
                if (likely(mask == 0))
                {
                    addr = node.children[i1]; dist = tnear[i1];
                    continue;
                }

                int i2 = bsf(mask);
                if (likely(mask == 0))
                {
                    if (tnear[i2] < tnear[i1]) std::swap(i2,i1);

                    stack[ptr++] = { node.children[i2], tnear[i2] };
                    addr = node.children[i1]; dist = tnear[i1];
                    continue;
                }

                int i3 = bsf(mask);
                if (likely(mask == 0))
                {
                    if (tnear[i2] < tnear[i1]) std::swap(i2,i1);
                    if (tnear[i3] < tnear[i2]) std::swap(i3,i2);
                    if (tnear[i3] < tnear[i1]) std::swap(i3,i1);

                    stack[ptr++] = { node.children[i3], tnear[i3] };
                    stack[ptr++] = { node.children[i2], tnear[i2] };
                    addr = node.children[i1]; dist = tnear[i1];
                    continue;
                }

                int i4 = bsf(mask);
                if (likely(mask == 0))
                {
                    if (tnear[i2] < tnear[i1]) std::swap(i2,i1);
                    if (tnear[i4] < tnear[i3]) std::swap(i4,i3);
                    if (tnear[i3] < tnear[i1]) std::swap(i3,i1);
                    if (tnear[i4] < tnear[i2]) std::swap(i4,i2);
                    if (tnear[i3] < tnear[i2]) std::swap(i3,i2);

                    stack[ptr++] = { node.children[i4], tnear[i4] };
                    stack[ptr++] = { node.children[i3], tnear[i3] };
                    stack[ptr++] = { node.children[i2], tnear[i2] };
                    addr = node.children[i1]; dist = tnear[i1];
                    continue;
                }

                if constexpr (BVH::Width > 4)
                {
                    char old = ptr;
                    stack[ptr++] = { node.children[i4], tnear[i4] };
                    stack[ptr++] = { node.children[i3], tnear[i3] };
                    stack[ptr++] = { node.children[i2], tnear[i2] };
                    stack[ptr++] = { node.children[i1], tnear[i1] };

                    do
                    {
                        int i = bsf(mask);
                        stack[ptr++] = { node.children[i], tnear[i] };
                    }
                    while (unlikely(mask != 0));

                    bubble_sort(stack + old, stack + ptr,
                        [](stack_entry const& s1, stack_entry const& s2) {
                            return s1.dist > s2.dist;
                        });

                    se = stack[--ptr];
                    addr = se.addr;
                    dist = se.dist;
                    continue;
                }
            }
            else if constexpr (Traversal == detail::AnyHit)
            {
                int i = bsf(mask);
                addr = node.children[i]; dist = tnear[i];

                while (unlikely(mask != 0))
                {
                    i = bsf(mask);
                    stack[ptr++] = { node.children[i], tnear[i] };
                }
                continue;
            }
        }

        // while node contains untested primitives
        //     perform a ray-primitive intersection test

        uint64_t first = addr.id;
        uint64_t num_prims = addr.num_prims;

        uint64_t last = first + num_prims;

        for (auto i = first; i != last; ++i)
        {
            auto prim = b.primitive(i);

            auto hr = HR(isect(ray, prim), i);
            auto closer = is_closer(hr, result, ray.tmin, ray.tmax);

            if (!closer)
            {
                continue;
            }

            update_if(result, hr, closer);

            if constexpr (Traversal == detail::AnyHit)
            {
                if (result.hit)
                {
                    return result;
                }
            }
        }
    }

    return result;
}
//-------------------------------------------------------------------------------------------------
// Default intersect returns closest hit!
//

// overload w/ custom intersector -------------------------

template <typename R, typename BVH, typename Intersector>
VSNRAY_FUNC
inline auto intersect_ray1_bvhN_compressed(
        R const&     ray,
        BVH const&   b,
        Intersector& isect
        )
    -> decltype(intersect_ray1_bvhN_compressed<detail::ClosestHit>(ray, b, isect))
{
    return intersect_ray1_bvhN_compressed<detail::ClosestHit>(ray, b, isect);
}

// overload w/ default intersector ------------------------

template <typename R, typename BVH>
VSNRAY_FUNC
inline auto intersect_ray1_bvhN_compressed(R const& ray, BVH const& b)
    -> decltype(intersect_ray1_bvhN_compressed<detail::ClosestHit>(
            ray,
            b,
            std::declval<default_intersector&>())
            )
{
    default_intersector isect;
    return intersect_ray1_bvhN_compressed<detail::ClosestHit>(ray, b, isect);
}

} // visionaray
