// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iterator>
#include <type_traits>
#include <utility>

#include <visionaray/bvh.h>
#include <visionaray/intersector.h>
#include <visionaray/update_if.h>

#include "exit_traversal.h"
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
        Intersector&                    isect
        )
{
    using RT = decltype(isect(r, *begin));

    RT result;

    for (P it = begin; it != end; ++it)
    {
        auto hr = isect(r, *it);
        update_if(result, hr, is_closer(hr, result, r.tmin, r.tmax));

        exit_traversal<Traversal> early_exit;
        if (early_exit.check(result))
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
        Intersector&                    isect
        )
{
    using RT = decltype( isect(
            std::integral_constant<int, Traversal>{},
            r,
            *begin
            ) );

    RT result;

    for (P it = begin; it != end; ++it)
    {
        auto hr = isect(
                std::integral_constant<int, Traversal>{},
                r,
                *it
                );

        update_if(result, hr, is_closer(hr, result, r.tmin, r.tmax));

        exit_traversal<Traversal> early_exit;
        if (early_exit.check(result))
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
// any hit
//

template <
    typename R,
    typename Primitives,
    typename Intersector,
    typename Primitive = typename std::iterator_traits<Primitives>::value_type
    >
VSNRAY_FUNC
inline auto any_hit(
        R const&        r,
        Primitives      begin,
        Primitives      end,
        Intersector&    isect
        )
{
    return detail::traverse<detail::AnyHit>(
            is_any_bvh<Primitive>{},
            r,
            begin,
            end,
            isect
            );
}

template <typename R, typename P>
VSNRAY_FUNC
inline auto any_hit(R const& r, P begin, P end)
{
    default_intersector ignore;
    return any_hit(r, begin, end, ignore);
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
{
    return detail::traverse<detail::ClosestHit>(
            is_any_bvh<Primitive>{},
            r,
            begin,
            end,
            isect
            );
}

template <typename R, typename Primitives>
VSNRAY_FUNC
inline auto closest_hit(R const& r, Primitives begin, Primitives end)
{
    default_intersector ignore;
    return closest_hit(r, begin, end, ignore);
}

} // visionaray
