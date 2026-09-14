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

inline void cmp_exchange(simd::int4 &a, simd::int4 &b)
{
    auto m0 = b < a;
    simd::mask4 m;
    m.i = simd::shuffle<2,2,2,2>(simd::int4(m0.i));
    simd::int4 c = select(m, b, a);
    simd::int4 d = select(m, a, b);
    a = c;
    b = d;
}

inline void sort(simd::int4& s0, simd::int4& s1, simd::int4& s2)
{
    cmp_exchange(s1, s0);
    cmp_exchange(s2, s1);
    cmp_exchange(s1, s0);
}

inline void sort(simd::int4& s0, simd::int4& s1, simd::int4& s2, simd::int4& s3)
{
    cmp_exchange(s1, s0);
    cmp_exchange(s3, s2);
    cmp_exchange(s2, s0);
    cmp_exchange(s3, s1);
    cmp_exchange(s2, s1);
}


namespace detail
{

template <typename HR>
inline HR closest(HR const& hr)
{
    return hr;
}

template <typename R, typename I, typename = std::enable_if_t<simd::is_simd_vector<I>::value>>
inline auto closest(hit_record<R, primitive<I>> const& hr)
{
    assert(any(hr.hit)); // !

    int closest_index = min_index(hr.t, hr.hit.i);

    hit_record<R, primitive<unsigned>> result;
    result.hit = true;
    result.prim_id = simd::get(hr.prim_id, closest_index);
    result.geom_id = simd::get(hr.geom_id, closest_index);
    result.inst_id = simd::get(hr.inst_id, closest_index);
    result.t = simd::get(hr.t, closest_index);
    result.isect_pos.x = simd::get(hr.isect_pos.x, closest_index);
    result.isect_pos.y = simd::get(hr.isect_pos.y, closest_index);
    result.isect_pos.z = simd::get(hr.isect_pos.z, closest_index);
    result.u = simd::get(hr.u, closest_index);
    result.v = simd::get(hr.v, closest_index);
    return result;
}

}


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
inline hit_record<R, primitive<unsigned>> intersect_ray1_bvhN(
        R const&     ray,
        BVH const&   b,
        Intersector& isect
        )
{
    using namespace detail;
    using HR = hit_record<R, primitive<unsigned>>;

    HR result;

    VSNRAY_ALIGN(16) struct stack_entry
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
                    if (tnear[i1] < tnear[i2])
                    {
                        stack[ptr++] = { node.children[i2], tnear[i2] };
                        addr = node.children[i1]; dist = tnear[i1];
                    }
                    else
                    {
                        stack[ptr++] = { node.children[i1], tnear[i1] };
                        addr = node.children[i2]; dist = tnear[i2];
                    }
                    continue;
                }

                int i3 = bsf(mask);
                if (likely(mask == 0))
                {
                    stack[ptr]     = { node.children[i1], tnear[i1] };
                    stack[ptr + 1] = { node.children[i2], tnear[i2] };
                    stack[ptr + 2] = { node.children[i3], tnear[i3] };
                    sort((simd::int4&)stack[ptr], (simd::int4&)stack[ptr + 1], (simd::int4&)stack[ptr + 2]);
                    ptr += 2;
                    se = stack[ptr];
                    addr = se.addr;
                    dist = se.dist;
                    continue;
                }

                int i4 = bsf(mask);
                if (likely(mask == 0))
                {
                    stack[ptr]     = { node.children[i1], tnear[i1] };
                    stack[ptr + 1] = { node.children[i2], tnear[i2] };
                    stack[ptr + 2] = { node.children[i3], tnear[i3] };
                    stack[ptr + 3] = { node.children[i4], tnear[i4] };
                    sort((simd::int4&)stack[ptr], (simd::int4&)stack[ptr + 1], (simd::int4&)stack[ptr + 2], (simd::int4&)stack[ptr + 3]);
                    ptr += 3;
                    se = stack[ptr];
                    addr = se.addr;
                    dist = se.dist;
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

        uint64_t first;
        uint64_t num_prims;

        bvh_multi_node<BVH::Width>::decode_leaf(addr, first, num_prims);

        auto prims = b.primitives() + first;

        for (uint64_t i = 0; i < num_prims; ++i)
        {
            auto hrN = isect(ray, prims[i]);

            if (!any(hrN.hit && hrN.t < r1.tmax))
            {
                continue;
            }

            result = detail::closest(hrN);
            r1.tmax = result.t;

            if constexpr (Traversal == detail::AnyHit)
            {
                return result;
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
