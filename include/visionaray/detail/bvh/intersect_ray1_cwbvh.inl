// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

//-------------------------------------------------------------------------------------------------
// Compressed-wide BVH traversal implementation after Ylitie et al.
// "Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs"
// This implementation was heavily inspired by https://github.com/AlanIWBFT/CWBVH
//

#include "cwbvh.h"

#ifdef __CUDA_ARCH__
#include <vector_types.h>
#endif

namespace visionaray
{

VSNRAY_FUNC
inline uint32_t find_first_set_bit(uint32_t val)
{
#ifdef __CUDA_ARCH__
    uint32_t ret;
    asm ("bfind.u32 %0, %1;" : "=r"(ret) : "r"(val));
    return ret;
#else
    return 31 ^ __builtin_clz(val);
#endif
}

VSNRAY_FUNC
inline uint32_t sign_replicate_s8x4(uint32_t v)
{
#ifdef __CUDA_ARCH__
    uint32_t i;
    asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(i) : "r"(v));
    return i;
#else
    uint32_t sign = v & 0x80808080;
    sign |= (sign >> 1);
    sign |= (sign >> 2);
    sign |= (sign >> 4);
    return sign;
#endif
}

VSNRAY_FUNC
inline uint32_t extract_byte(uint32_t val, uint32_t byteno)
{
    return (val >> (byteno * 8)) & 0xFF;
}

template <
    detail::traversal_type Traversal,
    typename R,
    typename BVH,
    typename Intersector
    >
VSNRAY_FUNC
inline hit_record<R, primitive<unsigned>> intersect_ray1_cwbvh(
        R            ray,
        BVH const&   b,
        Intersector& isect
        )
{
    hit_record<R, primitive<unsigned>> result;

    // Traversal stack
    vec2ui stack[32];
    char stack_ptr = 0;
    // Current node group
    vec2ui G(0, 1<<31); // root node (test first child first)

    uint32_t oct = ((*(uint32_t*)&ray.dir.z) & 0x80000000) >> 31
                 | ((*(uint32_t*)&ray.dir.y) & 0x80000000) >> 30
                 | ((*(uint32_t*)&ray.dir.x) & 0x80000000) >> 29;
    uint32_t octinv = 7 - oct;

    const float ooeps = exp2f(-80.0f);
    vec3 rcp_dir;//(
    rcp_dir.x = 1.0f / (fabsf(ray.dir.x) > ooeps ? ray.dir.x : copysignf(ooeps, ray.dir.x));
    rcp_dir.y = 1.0f / (fabsf(ray.dir.y) > ooeps ? ray.dir.y : copysignf(ooeps, ray.dir.y));
    rcp_dir.z = 1.0f / (fabsf(ray.dir.z) > ooeps ? ray.dir.z : copysignf(ooeps, ray.dir.z));

    for (;;)
    {
        vec2ui G_t(0u, 0u); // group that only consists of triangles
        if (G.y > 0x00FFFFFF)
        {
            // Get closest node
            const uint32_t imask = G.y;
            uint32_t child_bit = find_first_set_bit(G.y);
            const uint32_t child_node_base = G.x;

            // G <- G\n
            G.y &= ~(1 << child_bit);

            if (G.y > 0x00FFFFFF)
            {
                stack[stack_ptr++] = G;
            }

            // extract child node slot and load node from memory:
            uint32_t slot_index = (child_bit - 24) ^ octinv;
            uint32_t octinv4 = octinv * 0x01010101;
            uint32_t node_id = child_node_base + __builtin_popcount(imask & ~(0xFFFFFFFF << slot_index));

#ifdef __CUDA_ARCH__
            ::float4 const* nodes_base = (::float4 const*)b.nodes();
            cwbvh_node n;
            ::float4* nf4 = (::float4*)&n;
            nf4[0] = __ldg(nodes_base + node_id * 5 + 0);
            nf4[1] = __ldg(nodes_base + node_id * 5 + 1);
            nf4[2] = __ldg(nodes_base + node_id * 5 + 2);
            nf4[3] = __ldg(nodes_base + node_id * 5 + 3);
            nf4[4] = __ldg(nodes_base + node_id * 5 + 4);
#else
            auto const& n = b.node(node_id);
#endif

            // Transform ray:
            auto pow2 = [](char e) {
                unsigned u((e + 127) << 23);
                return *(float*)&u;
            };

            vec3 P(pow2(n.e[0]), pow2(n.e[1]), pow2(n.e[2]));
            vec3 local_ori((n.origin - ray.ori) * rcp_dir);
            vec3 local_dir(P * rcp_dir);

            // Set up node and triangle group base indices:
            G.x   = n.first_child;
            G_t.x = n.first_prim;
            G_t.y = 0;
            uint32_t hitmask = 0u;

            // Intersect first pack of four:
            {
                uint32_t meta4       = *(uint32_t*)&n.meta[0];
                uint32_t is_inner4   = (meta4 & (meta4 << 1)) & 0x10101010;
                uint32_t inner_mask4 = sign_replicate_s8x4(is_inner4 << 3);
                uint32_t bit_index4  = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
                uint32_t child_bits4 = (meta4 >> 5) & 0x07070707;

                for (int i = 0; i < 4; ++i)
                {
                    const vec3 lo = vec3(n.minx[i + 0], n.miny[i + 0], n.minz[i + 0]);
                    const vec3 hi = vec3(n.maxx[i + 0], n.maxy[i + 0], n.maxz[i + 0]);

                    const vec3 t_lo = local_ori + lo * local_dir;
                    const vec3 t_hi = local_ori + hi * local_dir;

                    const vec3 t_nr = min(t_lo, t_hi);
                    const vec3 t_fr = max(t_lo, t_hi);

                    const float tnear = max(ray.tmin, max_element(t_nr));
                    const float tfar = min(ray.tmax, min_element(t_fr));

                    bool intersected = tfar >= tnear;

                    if (intersected)
                    {
                        uint32_t child_bits = extract_byte(child_bits4, i);
                        uint32_t bit_index = extract_byte(bit_index4, i);
                        hitmask |= child_bits << bit_index;
                    }
                }
            }

            // Intersect second pack of four:
            {
                uint32_t meta4       = *(uint32_t*)&n.meta[4];
                uint32_t is_inner4   = (meta4 & (meta4 << 1)) & 0x10101010;
                uint32_t inner_mask4 = sign_replicate_s8x4(is_inner4 << 3);
                uint32_t bit_index4  = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
                uint32_t child_bits4 = (meta4 >> 5) & 0x07070707;

                for (int i = 0; i < 4; ++i)
                {
                    const vec3 lo = vec3(n.minx[i + 4], n.miny[i + 4], n.minz[i + 4]);
                    const vec3 hi = vec3(n.maxx[i + 4], n.maxy[i + 4], n.maxz[i + 4]);

                    const vec3 t_lo = local_ori + lo * local_dir;
                    const vec3 t_hi = local_ori + hi * local_dir;

                    const vec3 t_nr = min(t_lo, t_hi);
                    const vec3 t_fr = max(t_lo, t_hi);

                    const float tnear = max(ray.tmin, max_element(t_nr));
                    const float tfar = min(ray.tmax, min_element(t_fr));

                    bool intersected = tfar >= tnear;

                    if (intersected)
                    {
                        uint32_t child_bits = extract_byte(child_bits4, i);
                        uint32_t bit_index = extract_byte(bit_index4, i);
                        hitmask |= child_bits << bit_index;
                    }
                }
            }

            // Patch hitmask into internal and triangle groups:
            G.y   = (hitmask & 0xFF000000) | n.imask;
            G_t.y =  hitmask & 0x00FFFFFF;
        }
        else // G represents triangle group
        {
            G_t = G;
            G = vec2ui(0, 0);
        }

#ifdef __CUDA_ARCH__
        const int totalThreads = __popc(__activemask());
#endif

        while (G_t.y != 0)
        {
#ifdef __CUDA_ARCH__
            const float Rt = 0.2;
            const int threshold = totalThreads * Rt;
            const int numActiveThreads = __popc(__activemask());
            if (numActiveThreads < threshold)
            {
                stack[stack_ptr++] = G_t;
                break;
            }
#endif

            int prim_index_relative = find_first_set_bit(G_t.y);
            int prim_id = G_t.x + prim_index_relative;
            auto const& prim = b.primitive(prim_id);

            auto hr = isect(ray, prim);
            if (hr.hit && hr.t < ray.tmax)
            {
                result = hr;

                if constexpr (Traversal == detail::AnyHit)
                {
                    return result;
                }

                ray.tmax = result.t;
            }

            // remove from G_t
            G_t.y &= ~(1 << prim_index_relative);
        }

        if (G.y <= 0x00FFFFFF)
        {
            if (stack_ptr == 0)
            {
                break;
            }

            G = stack[--stack_ptr];
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
inline auto intersect_ray1_cwbvh(
        R const&     ray,
        BVH const&   b,
        Intersector& isect
        )
    -> decltype(intersect_ray1_cwbvh<detail::ClosestHit>(ray, b, isect))
{
    return intersect_ray1_cwbvh<detail::ClosestHit>(ray, b, isect);
}

// overload w/ default intersector ------------------------

template <typename R, typename BVH>
VSNRAY_FUNC
inline auto intersect_ray1_cwbvh(R const& ray, BVH const& b)
    -> decltype(intersect_ray1_cwbvh<detail::ClosestHit>(
            ray,
            b,
            std::declval<default_intersector&>())
            )
{
    default_intersector isect;
    return intersect_ray1_cwbvh<detail::ClosestHit>(ray, b, isect);
}

} // visionaray
