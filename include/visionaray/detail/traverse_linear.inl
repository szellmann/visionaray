// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <utility>

#include <visionaray/intersector.h>

#include "macros.h"

namespace visionaray
{
namespace detail
{

template <bool AnyHit, typename R, typename P, typename Intersector>
VSNRAY_FUNC
auto traverse(
        R const& r,
        P begin,
        P end,
        typename R::scalar_type const& max_t,
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

template <bool AnyHit, typename R, typename P, typename Intersector>
VSNRAY_FUNC
auto traverse(
        R const& r,
        P begin,
        P end,
        Intersector& isect
        )
    -> decltype( isect(r, *begin) )
{
    return traverse<AnyHit>(r, begin, end, numeric_limits<float>::max(), isect);
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
    return detail::traverse<true>(r, begin, end, isect);
}

template <typename R, typename P>
VSNRAY_FUNC
auto any_hit(R const& r, P begin, P end)
    -> decltype( intersect(r, *begin) )
{
    default_intersector ignore;
    return detail::traverse<true>(r, begin, end, ignore);
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
    return detail::traverse<true>(r, begin, end, max_t, isect);
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
    return detail::traverse<true>(r, begin, end, max_t, ignore);
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
    return detail::traverse<false>(r, begin, end, isect);
}

template <typename R, typename P>
VSNRAY_FUNC
auto closest_hit(R const& r, P begin, P end)
    -> decltype( std::declval<default_intersector>()(r, *begin) )
{
    default_intersector ignore;
    return detail::traverse<false>(r, begin, end, ignore);
}

} // visionaray
