// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iterator>
#include <utility>
#include <type_traits>

#include <visionaray/intersector.h>
#include <visionaray/update_if.h>

#include "macros.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Traverse any primitives but BVHs
//

template <
    traversal_type Traversal,
    typename R,
    typename P,
    typename Intersector
    >
VSNRAY_FUNC
inline auto traverse(
        std::false_type                 /* is no bvh */,
        R const&                        r,
        P                               begin,
        P                               end,
        typename R::scalar_type const&  max_t,
        Intersector& isect
        )
    -> decltype( isect(r, *begin) )
{
    using HR = decltype( isect(r, *begin) );

    HR result;

    for (P it = begin; it != end; ++it)
    {
        auto hr = isect(r, *it);
        update_if(result, hr, is_closer(hr, result, max_t));

        if ( Traversal == AnyHit && all(result.hit) )
        {
            return result;
        }
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Traverse BVHs. Those have a special intersect method that takes the traversal
// type as an additional parameter.
//

template <
    traversal_type Traversal,
    typename R,
    typename P,
    typename Intersector
    >
VSNRAY_FUNC
inline auto traverse(
        std::true_type                  /* is_bvh */,
        R const&                        r,
        P                               begin,
        P                               end,
        typename R::scalar_type const&  max_t,
        Intersector& isect
        )
    -> decltype( isect(std::integral_constant<int, Traversal>{}, r, *begin, max_t) )
{
    using HR = decltype( isect(std::integral_constant<int, Traversal>{}, r, *begin, max_t) );

    HR result;

    for (P it = begin; it != end; ++it)
    {
        auto hr = isect(std::integral_constant<int, Traversal>{}, r, *it, max_t);
        update_if(result, hr, is_closer(hr, result, max_t));

        if ( Traversal == AnyHit && all(result.hit) )
        {
            return result;
        }
    }

    return result;
}

template <
    traversal_type Traversal,
    typename IsAnyBVH,
    typename R,
    typename Primitives,
    typename Intersector
    >
VSNRAY_FUNC
inline auto traverse(
        IsAnyBVH        /* */,
        R const&        r,
        Primitives      begin,
        Primitives      end,
        Intersector&    isect
        )
    -> decltype( traverse<Traversal>(
            IsAnyBVH{},
            r,
            begin,
            end,
            numeric_limits<float>::max(),
            isect
            ) )
{
    return traverse<Traversal>(
            IsAnyBVH{},
            r,
            begin,
            end,
            numeric_limits<float>::max(),
            isect
            );
}

} // detail



//-------------------------------------------------------------------------------------------------
// any hit w/o max_t
//

template <typename R, typename P, typename Intersector>
VSNRAY_FUNC
inline auto any_hit(
        R const& r,
        P begin,
        P end,
        Intersector& isect
        )
    -> decltype( isect(r, *begin) )
{
    using Primitive = typename std::iterator_traits<P>::value_type;

    return detail::traverse<detail::AnyHit>(
            std::integral_constant<bool, is_any_bvh<Primitive>::value>{},
            r,
            begin,
            end,
            isect
            );
}

template <typename R, typename P>
VSNRAY_FUNC
inline auto any_hit(R const& r, P begin, P end)
    -> decltype( intersect(r, *begin) )
{
    default_intersector ignore;
    return any_hit(r, begin, end, ignore);
}


//-------------------------------------------------------------------------------------------------
// any hit with max_t
//

template <
    typename R,
    typename Primitives,
    typename Intersector,
    typename Primitive = typename std::iterator_traits<Primitives>::value_type
    >
VSNRAY_FUNC
inline auto any_hit(
        R const&                        r,
        Primitives                      begin,
        Primitives                      end,
        typename R::scalar_type const&  max_t,
        Intersector&                    isect
        )
    -> decltype( detail::traverse<detail::AnyHit>(
            std::integral_constant<bool, is_any_bvh<Primitive>::value>{},
            r,
            begin,
            end,
            max_t,
            isect
            ) )
{
    return detail::traverse<detail::AnyHit>(
            std::integral_constant<bool, is_any_bvh<Primitive>::value>{},
            r,
            begin,
            end,
            max_t,
            isect
            );
}

template <typename R, typename Primitives>
VSNRAY_FUNC
inline auto any_hit(
        R const&                        r,
        Primitives                      begin,
        Primitives                      end,
        typename R::scalar_type const&  max_t
        )
    -> decltype( any_hit(r, begin, end, max_t, std::declval<default_intersector&>() ) )
{
    default_intersector ignore;
    return any_hit(r, begin, end, max_t, ignore);
}


//-------------------------------------------------------------------------------------------------
// closest hit 
//

template <
    typename R,
    typename Primitives,
    typename Intersector,
    typename Primitive = typename std::iterator_traits<Primitives>::value_type
    >
VSNRAY_FUNC
inline auto closest_hit(
        R const&        r,
        Primitives      begin,
        Primitives      end,
        Intersector&    isect
        )
    -> decltype( detail::traverse<detail::ClosestHit>(
            std::integral_constant<bool, is_any_bvh<Primitive>::value>{},
            r,
            begin,
            end,
            isect
            ) )
{
    return detail::traverse<detail::ClosestHit>(
            std::integral_constant<bool, is_any_bvh<Primitive>::value>{},
            r,
            begin,
            end,
            isect
            );
}

template <typename R, typename Primitives>
VSNRAY_FUNC
inline auto closest_hit(R const& r, Primitives begin, Primitives end)
    -> decltype( closest_hit(r, begin, end, std::declval<default_intersector&>()) )
{
    default_intersector ignore;
    return closest_hit(r, begin, end, ignore);
}

} // visionaray
