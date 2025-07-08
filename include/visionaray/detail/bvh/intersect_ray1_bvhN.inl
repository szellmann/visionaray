// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <visionaray/math/simd/type_traits.h>
#include <visionaray/math/matrix.h>
#include <visionaray/intersector.h>
#include <visionaray/update_if.h>

#include "../tags.h"
#include "hit_record.h"

#ifdef _MSC_VER
// TODO:
#define likely(x) x
#define unlikely(x) x
#include <intrin.h>
inline unsigned ctz(unsigned v)
{
    unsigned long tz = 0;
    if (_BitScanForward(&tz, v))
    {
        return tz;
    }
    else
    {
        return 32u;
    }
}
#else
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define ctz(x) __builtin_ctz(x)
#endif

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// pull out ray/box intersection to safe us some conversions:
//

template <typename F>
struct hit_record_ray1_boxN
{
    using M = simd::mask_from_simd_width_t<simd::num_elements<F>::value>;

    F tnear;
    F tfar;
    M hit;
};

template <typename F>
struct ray1
{
    vector<3, F> ori;
    vector<3, F> inv_dir;
    F tmin;
    F tmax;
};

template <typename F>
inline ray1<F> make_ray1(basic_ray<float> const& r)
{
    vec3 inv_dir;
    inv_dir.x = r.dir.x != 0.0f ? 1.0f / r.dir.x : FLT_MAX;
    inv_dir.y = r.dir.y != 0.0f ? 1.0f / r.dir.y : FLT_MAX;
    inv_dir.z = r.dir.z != 0.0f ? 1.0f / r.dir.z : FLT_MAX;

    ray1<F> res;
    res.ori = vector<3, F>(r.ori);
    res.inv_dir = vector<3, F>(inv_dir);
    res.tmin = F(r.tmin);
    res.tmax = F(r.tmax);
    return res;
}

template <typename F>
MATH_FUNC
inline hit_record_ray1_boxN<F> intersect_ray1_boxN(ray1<F> const& r, basic_aabb<F> const& aabb)
{
    hit_record_ray1_boxN<F> result;

    vector<3, F> t1 = (aabb.min - r.ori) * r.inv_dir;
    vector<3, F> t2 = (aabb.max - r.ori) * r.inv_dir;

    vector<3, F> tmin = min(t1, t2);
    vector<3, F> tmax = max(t1, t2);

    result.tnear = max(r.tmin, max(tmin.x, max(tmin.y, tmin.z)));
    result.tfar  = min(r.tmax, min(tmax.x, min(tmax.y, tmax.z)));

    // validity check:
    result.hit = aabb.min.x <= aabb.max.x;

    result.hit &= result.tfar >= result.tnear;

    return result;
}

template <typename It, typename Comp>
inline void bubble_sort(It first, It last, Comp comp)
{
    int n = last - first;

    for (int i = 0; i < n - 1; ++i)
    {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (comp(first[j + 1], first[j]))
            {
                auto temp = first[j];
                first[j] = first[j + 1];
                first[j + 1] = temp;
                swapped = true;
            }
        }

        if (!swapped)
        {
            break;
        }
    }
}

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_NEON_FP)

// From SSE2Neon:
inline int movemask(uint32x4_t const& input)
{
    static const int32_t shift[4] = {0, 1, 2, 3};
    uint32x4_t tmp = vshrq_n_u32(input, 31);
    return vaddvq_u32(vshlq_u32(tmp, vld1q_s32(shift)));
}

inline int movemask(uint32x4_t const input[2])
{
    return (movemask(input[1]) << 4) | movemask(input[0]);
}

#elif VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE2)

inline int movemask(__m128i const& input)
{
    return _mm_movemask_ps(_mm_castsi128_ps(input));
}

inline int movemask(__m256i const& input)
{
    return _mm256_movemask_ps(_mm256_castsi256_ps(input));
}

#endif

//-----------------------------------------------------------------------------
// SSE and NEON traversal based on:
// https://afra.dev/publications/Afra2013Incoherent.pdf
//

template <
    detail::traversal_type Traversal,
    typename R,
    typename BVH,
    typename Intersector,
    typename T = typename R::scalar_type
    >
VSNRAY_FUNC
inline auto intersect_ray1_bvhN(
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

    struct stack_entry
    {
        int64_t addr;
        unsigned dist;
    };

    stack_entry stack[64];
    char ptr = 0;
    stack[ptr++] = { 0, 0 }; // root node

    using F = simd::float_from_simd_width_t<BVH::Width>;

    auto r1 = make_ray1<F>(ray);

    // while ray not terminated
next:
    while (ptr > 0)
    {
        auto se = stack[--ptr];
        int64_t addr = se.addr;
        unsigned dist = se.dist;

        // while node does not contain primitives
        //     traverse to the next node

        while (addr >= 0)
        {
            if (*((unsigned*)&result.t) < dist)
            {
                goto next;
            }

            const auto &node = b.node(addr);

            basic_aabb<F> aabbN;
            node.bounds_as_floatN(aabbN);

            auto hrN = intersect_ray1_boxN(r1, aabbN);

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_NEON_FP)
            hrN.hit &= reinterpret_as_uint(hrN.tnear) < reinterpret_as_uint(F(result.t));
#else
            // TODO: unsigned comparisons with SSE, and check if this is worth it..
            hrN.hit &= hrN.tnear < F(result.t);
#endif

            auto mask = movemask(hrN.hit.i);

            if (!mask)
            {
                goto next;
            }

            unsigned* tnear = reinterpret_cast<unsigned*>(&hrN.tnear);

#if 1
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
#else
            // Unoptimized code path, keeping this around for the
            // moment so we can compare:

            unsigned child_count = node.get_num_children();

            unsigned* hit = reinterpret_cast<unsigned*>(&hrN.hit);

            int idx[BVH::Width];
            for (int i = 0; i < BVH::Width; ++i)
            {
                idx[i] = i;
            }

            bubble_sort(idx, idx + child_count,
                [&](int i, int j) {
                    return (hit[i] && hit[j] && tnear[i] < tnear[j]) ||
                           (hit[i] && !hit[j]);
                });

            for (int i = 1; i < child_count; ++i)
            {
                if (!hit[idx[i]])
                {
                    break;
                }

                stack[ptr++] = node.children[idx[i]];
            }

            addr = node.children[idx[0]];
#endif
        }

        // while node contains untested primitives
        //     perform a ray-primitive intersection test

        uint64_t first;
        uint64_t num_prims;

        bvh_multi_node<BVH::Width>::decode_leaf(addr, first, num_prims);

        uint64_t last = first + num_prims;

        for (auto i = first; i != last; ++i)
        {
            auto prim = b.primitive(i);

            auto hr = HR(isect(ray, prim), i);
            auto closer = is_closer(hr, result, ray.tmin, ray.tmax);

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
inline auto intersect_ray1_bvhN(
        R const&     ray,
        BVH const&   b,
        Intersector& isect
        )
    -> decltype(intersect_ray1_bvhN<detail::ClosestHit>(ray, b, isect))
{
    return intersect_ray1_bvhN<detail::ClosestHit>(ray, b, isect);
}

// overload w/ default intersector ------------------------

template <typename R, typename BVH>
VSNRAY_FUNC
inline auto intersect_ray1_bvhN(R const& ray, BVH const& b)
    -> decltype(intersect_ray1_bvhN<detail::ClosestHit>(
            ray,
            b,
            std::declval<default_intersector&>())
            )
{
    default_intersector isect;
    return intersect_ray1_bvhN<detail::ClosestHit>(ray, b, isect);
}

} // visionaray
