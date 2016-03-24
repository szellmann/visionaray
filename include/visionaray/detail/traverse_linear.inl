// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iterator>
#include <utility>
#include <type_traits>

#include <visionaray/intersector.h>

#include "macros.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Traverse any primitives but BVHs
//

template <
    bool AnyHit,
    typename R,
    typename P,
    typename Intersector
    >
VSNRAY_FUNC
auto traverse(
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

        if ( AnyHit && all(result.hit) )
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
    bool AnyHit,
    typename R,
    typename P,
    typename Intersector
    >
VSNRAY_FUNC
auto traverse(
        std::true_type                  /* is_bvh */,
        R const&                        r,
        P                               begin,
        P                               end,
        typename R::scalar_type const&  max_t,
        Intersector& isect
        )
    -> decltype( isect(std::integral_constant<bool, AnyHit>{}, r, *begin) )
{
    using HR = decltype( isect(r, *begin) );

    HR result;

    for (P it = begin; it != end; ++it)
    {
        auto hr = isect(std::integral_constant<bool, AnyHit>{}, r, *it);
        update_if(result, hr, is_closer(hr, result, max_t));

        if ( AnyHit && all(result.hit) )
        {
            return result;
        }
    }

    return result;
}

template <
    bool AnyHit,
    typename IsAnyBVH,
    typename R,
    typename P,
    typename Intersector
    >
VSNRAY_FUNC
auto traverse(
        IsAnyBVH        /* */,
        R const&        r,
        P               begin,
        P               end,
        Intersector&    isect
        )
    -> decltype( isect(r, *begin) )
{
    return traverse<AnyHit>(
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
auto any_hit(
        R const& r,
        P begin,
        P end,
        Intersector& isect
        )
    -> decltype( isect(r, *begin) )
{
    using Primitive = typename std::iterator_traits<P>::value_type;

    return detail::traverse<true>(
            std::integral_constant<bool, is_any_bvh<Primitive>::value>{},
            r,
            begin,
            end,
            isect
            );
}

template <typename R, typename P>
VSNRAY_FUNC
auto any_hit(R const& r, P begin, P end)
    -> decltype( intersect(r, *begin) )
{
    default_intersector ignore;
    return any_hit(r, begin, end, ignore);
}


//-------------------------------------------------------------------------------------------------
// any hit with max_t
//

template <typename R, typename P, typename Intersector>
VSNRAY_FUNC
auto any_hit(
        R const& r,
        P begin,
        P end,
        typename R::scalar_type const& max_t,
        Intersector& isect
        )
    -> decltype( isect(r, *begin) )
{
    using Primitive = typename std::iterator_traits<P>::value_type;

    return detail::traverse<true>(
            std::integral_constant<bool, is_any_bvh<Primitive>::value>{},
            r,
            begin,
            end,
            max_t,
            isect
            );
}

template <typename R, typename P>
VSNRAY_FUNC
auto any_hit(
        R const& r,
        P begin,
        P end,
        typename R::scalar_type const& max_t
        )
    -> decltype( std::declval<default_intersector>()(r, *begin) )
{
    default_intersector ignore;
    return any_hit(r, begin, end, max_t, ignore);
}


//-------------------------------------------------------------------------------------------------
// closest hit 
//

template <typename R, typename P, typename Intersector>
VSNRAY_FUNC
auto closest_hit(
        R const& r,
        P begin,
        P end,
        Intersector& isect
        )
    -> decltype( isect(r, *begin) )
{
    using Primitive = typename std::iterator_traits<P>::value_type;

    return detail::traverse<false>(
            std::integral_constant<bool, is_any_bvh<Primitive>::value>{},
            r,
            begin,
            end,
            isect
            );
}

template <typename R, typename P>
VSNRAY_FUNC
auto closest_hit(R const& r, P begin, P end)
    -> decltype( std::declval<default_intersector>()(r, *begin) )
{
    default_intersector ignore;
    return closest_hit(r, begin, end, ignore);
}

} // visionaray
