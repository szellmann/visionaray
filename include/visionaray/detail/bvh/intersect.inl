// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <type_traits>
#include <utility>

#include <visionaray/math/simd/type_traits.h>
#include <visionaray/math/matrix.h>
#include <visionaray/intersector.h>
#include <visionaray/update_if.h>

#include "../exit_traversal.h"
#include "../stack.h"
#include "../tags.h"
#include "hit_record.h"

#if defined( __CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define VSNRAY_FULL_STACK_TRAVERSAL_ 0
#else
#define VSNRAY_FULL_STACK_TRAVERSAL_ 1
#endif

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Ray / BVH intersection
//

template <
    detail::traversal_type Traversal,
    typename R,
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename = typename std::enable_if<!is_any_bvh_inst<BVH>::value>::type,
    typename Intersector,
    typename T = typename R::scalar_type
    >
VSNRAY_FUNC
inline auto intersect(
        R const&     ray,
        BVH const&   b,
        Intersector& isect
        )
    -> hit_record_bvh<
            R,
            decltype( isect(ray, std::declval<typename BVH::primitive_type>()) )
            >
{
#if VSNRAY_FULL_STACK_TRAVERSAL_
    using namespace detail;
    using HR = hit_record_bvh<R, decltype(isect(ray, std::declval<typename BVH::primitive_type>()))>;

    using I = typename simd::int_type_t<T>;
    using M = typename simd::mask_type_t<T>;

    HR result;

    stack<32> st;
    st.push(0); // address of root node

    vector<3, T> inv_dir(
        select(ray.dir.x != T(0.0), T(1.0) / ray.dir.x, T(0.0)),
        select(ray.dir.y != T(0.0), T(1.0) / ray.dir.y, T(0.0)),
        select(ray.dir.z != T(0.0), T(1.0) / ray.dir.z, T(0.0))
        );

    // while ray not terminated
next:
    while (!st.empty())
    {
        auto node = b.node(st.pop());

        if (simd::is_simd_vector<T>::value)
        {
            // while node does not contain primitives
            //     traverse to the next node

            while (!is_leaf(node))
            {
                auto children = &b.node(node.get_child(0));

                auto hr1 = isect(ray, children[0].get_bounds(), inv_dir);
                auto hr2 = isect(ray, children[1].get_bounds(), inv_dir);

                auto b1 = any(is_closer(hr1, result, ray.tmin, ray.tmax));
                auto b2 = any(is_closer(hr2, result, ray.tmin, ray.tmax));

                if (b1 && b2) 
                {   
                    unsigned near_addr = all( hr1.tnear < hr2.tnear ) ? 0 : 1;
                    st.push(node.get_child(!near_addr));
                    node = b.node(node.get_child(near_addr));
                }   
                else if (b1)
                {   
                    node = b.node(node.get_child(0));
                }   
                else if (b2)
                {   
                    node = b.node(node.get_child(1));
                }   
                else
                {   
                    goto next;
                }   
            }
        }
        else
        {
            while (true)
            {
                auto hr = isect(ray, node.get_bounds(), inv_dir);
                auto hit = any(is_closer(hr, result, ray.tmin, ray.tmax));

                if (!hit)
                {
                    goto next;
                }

                if (node.is_leaf())
                {
                    break;
                }

                I sign((int)node.ordered_traversal_sign);
                I sign_rd = reinterpret_as_int(ray.dir[node.ordered_traversal_axis]) >> 31;
                unsigned near_addr = any(M(sign ^ sign_rd));
                unsigned far_addr = !near_addr;

                st.push(node.get_child(far_addr));
                node = b.node(node.get_child(near_addr));
            }
        }


        // while node contains untested primitives
        //     perform a ray-primitive intersection test

        for (auto i = node.get_indices().first; i != node.get_indices().last; ++i)
        {
            auto prim = b.primitive(i);

            auto hr = HR(isect(ray, prim), i);
            auto closer = is_closer(hr, result, ray.tmin, ray.tmax);

#ifndef __CUDA_ARCH__
            if (!any(closer))
            {
                continue;
            }
#endif

            update_if(result, hr, closer);

            exit_traversal<Traversal> early_exit;
            if (early_exit.check(result))
            {
                return result;
            }
        }
    }

    return result;
#else
    using namespace detail;
    using HR = hit_record_bvh<R, decltype(isect(ray, std::declval<typename BVH::primitive_type>()))>;

    HR result;

    auto inv_dir = T(1.0) / ray.dir;

    bvh_node node = b.node(0);

    uint64_t level = 0x8000000000000000ULL;

    uint64_t pop_level = uint64_t(-1);

    uint64_t trail = 0;

    unsigned st1 = unsigned(-1);
    unsigned st2 = unsigned(-1);
    unsigned st3 = unsigned(-1);
    unsigned st4 = unsigned(-1);
    unsigned st5 = unsigned(-1);

    auto push_st = [&](unsigned n)
    {
        st1 = st2;
        st2 = st3;
        st3 = st4;
        st4 = st5;
        st5 = n;
    };

    auto pop_st = [&]()
    {
        unsigned res = st5;
        st5 = st4;
        st4 = st3;
        st3 = st2;
        st2 = st1;
        st1 = unsigned(-1);
        return res;
    };

    auto empty_st = [&st5]()
    {
        return st5 == unsigned(-1);
    };

    auto push = [&](unsigned n)
    {
        return push_st(n);
    };

    auto pop = [&]()
        -> bool
    {
        trail &= -level;
        trail += level;

        uint64_t temp = trail >> 1;
        level = ((temp - 1) ^ temp) + 1;

        if (trail & 0x8000000000000000ULL)
        {
            return true;
        }

        pop_level = level;

        if (empty_st())
        {
            node = b.node(0);
            level = 0x8000000000000000ULL;
        }
        else
        {
            node = b.node(pop_st());
        }

        return false;
    };

    while (true)
    {
        while (!is_leaf(node))
        {
            auto children = b.nodes() + node.get_child(0);

            aabb box1 = children[0].get_bounds();
            aabb box2 = children[1].get_bounds();

            const vec3 t_lo1 = (box1.min - ray.ori) * inv_dir;
            const vec3 t_hi1 = (box1.max - ray.ori) * inv_dir;

            const vec3 t_lo2 = (box2.min - ray.ori) * inv_dir;
            const vec3 t_hi2 = (box2.max - ray.ori) * inv_dir;

            const vec3 t_nr1 = min(t_lo1, t_hi1);
            const vec3 t_nr2 = min(t_lo2, t_hi2);

            const vec3 t_fr1 = max(t_lo1, t_hi1);
            const vec3 t_fr2 = max(t_lo2, t_hi2);

            const float tnear1 = max(ray.tmin, max_element(t_nr1));
            const float tnear2 = max(ray.tmin, max_element(t_nr2));

            const float tfar1 = min(ray.tmax, min_element(t_fr1));
            const float tfar2 = min(ray.tmax, min_element(t_fr2));

            const bool b1 = tfar1 >= tnear1 && tnear1 < result.t;
            const bool b2 = tfar2 >= tnear2 && tnear2 < result.t;

            if (b1 && b2)
            {
                unsigned near_addr = tnear1 < tnear2 ? 0 : 1;
                unsigned far_addr = !near_addr;
                level >>= 1;
                if ((trail & level) != 0)
                {
                    node = children[far_addr];
                }
                else
                {
                    push(node.get_child(far_addr));
                    node = children[near_addr];
                }
            }
            else if (b1 || b2)
            {
                level >>= 1;
                if (level != pop_level)
                {
                    trail |= level;
                    node = b1 ? children[0] : children[1];
                }
                else
                {
                    bool end_traversal = pop();
                    if (end_traversal)
                    {
                        return result;
                    }
                }
            }
            else
            {
                bool end_traversal = pop();
                if (end_traversal)
                {
                    return result;
                }
            }
        }


        // while node contains untested primitives
        //     perform a ray-primitive intersection test

        for (auto i = node.get_indices().first; i != node.get_indices().last; ++i)
        {
            auto prim = b.primitive(i);

            auto hr = HR(isect(ray, prim), i);
            auto closer = is_closer(hr, result, ray.tmin, ray.tmax);

#ifndef __CUDA_ARCH__
            if (!any(closer))
            {
                continue;
            }
#endif

            update_if(result, hr, closer);

            exit_traversal<Traversal> early_exit;
            if (early_exit.check(result))
            {
                return result;
            }
        }

        bool end_traversal = pop();
        if (end_traversal)
        {
            return result;
        }
    }
#endif
}


// Overload for instances ---------------------------------

template <
    detail::traversal_type Traversal,
    typename R,
    typename BVH,
    typename = typename std::enable_if<is_any_bvh_inst<BVH>::value>::type,
    typename Intersector,
    typename T = typename R::scalar_type
    >
VSNRAY_FUNC
inline auto intersect(
        R const&     ray,
        BVH const&   b,
        Intersector& isect
        )
    -> hit_record_bvh_inst<
            R,
            decltype( isect(ray, std::declval<typename BVH::primitive_type>()) )
            >
{
    using namespace detail;
    using HR = hit_record_bvh_inst<R, decltype(isect(ray, std::declval<typename BVH::primitive_type>()))>;

    R transformed_ray = ray;
    b.transform_ray(transformed_ray);

    auto hr = intersect<Traversal>(
            transformed_ray,
            b.get_ref(),
            isect
            );

    return HR(hr, hr.primitive_list_index, b.get_inst_id());
}


//-------------------------------------------------------------------------------------------------
// Default intersect returns closest hit!
//

// overload w/ custom intersector -------------------------

template <
    typename R,
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename Intersector
    >
VSNRAY_FUNC
inline auto intersect(
        R const&     ray,
        BVH const&   b,
        Intersector& isect
        )
    -> decltype(intersect<detail::ClosestHit>(ray, b, isect))
{
    return intersect<detail::ClosestHit>(ray, b, isect);
}

// overload w/ default intersector ------------------------

template <
    typename R,
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type
    >
VSNRAY_FUNC
inline auto intersect(R const& ray, BVH const& b)
    -> decltype(intersect<detail::ClosestHit>(
            ray,
            b,
            std::declval<default_intersector&>())
            )
{
    default_intersector isect;
    return intersect<detail::ClosestHit>(ray, b, isect);
}


} // visionaray

#undef VSNRAY_FULL_STACK_TRAVERSAL_
