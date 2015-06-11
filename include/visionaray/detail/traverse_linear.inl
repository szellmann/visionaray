// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/intersector.h>

#include "macros.h"

namespace visionaray
{
namespace detail
{

template <bool AnyHit, typename R, typename P, typename Intersector>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> traverse(
        R const& r,
        P begin,
        P end,
        typename R::scalar_type max_t,
        Intersector& isect
        )
{

    typedef typename R::scalar_type scalar_type;
    typedef P prim_iterator;

    hit_record<R, primitive<unsigned>> result;
    result.hit = false;
    result.t = numeric_limits<float>::max();
    result.prim_id = 0;

    for (prim_iterator it = begin; it != end; ++it)
    {
        auto hr = isect(r, *it);
        auto closer = hr.hit & ( hr.t >= scalar_type(0.0) && hr.t < max_t && hr.t < result.t );
        result.hit |= closer;
        result.t = select( closer, hr.t, result.t );
        result.prim_type = select( closer, hr.prim_type, result.prim_type );
        result.prim_id   = select( closer, hr.prim_id, result.prim_id );
        result.geom_id   = select( closer, hr.geom_id, result.geom_id );
        result.u         = select( closer, hr.u, result.u );
        result.v         = select( closer, hr.v, result.v );

        if ( AnyHit && all(result.hit) )
        {
            return result;
        }
    }

    return result;

}

template <bool AnyHit, typename R, typename P, typename Intersector>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> traverse(
        R const& r,
        P begin,
        P end,
        Intersector& isect)
{
    return traverse<AnyHit>(r, begin, end, numeric_limits<float>::max(), isect);
}

} // detail



//-------------------------------------------------------------------------------------------------
// any hit w/o max_t
//

template <typename R, typename P, typename Intersector>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> any_hit(
        R const& r,
        P begin,
        P end,
        Intersector& isect
        )
{
    return detail::traverse<true>(r, begin, end, isect);
}

template <typename R, typename P>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> any_hit(R const& r, P begin, P end)
{
    default_intersector ignore;
    return detail::traverse<true>(r, begin, end, ignore);
}


//-------------------------------------------------------------------------------------------------
// any hit with max_t
//

template <typename R, typename P, typename Intersector>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> any_hit(
        R const& r,
        P begin,
        P end,
        typename R::scalar_type max_t,
        Intersector& isect
        )
{
    return detail::traverse<true>(r, begin, end, max_t, isect);
}

template <typename R, typename P>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> any_hit(
        R const& r,
        P begin,
        P end,
        typename R::scalar_type max_t
        )
{
    default_intersector ignore;
    return detail::traverse<true>(r, begin, end, max_t, ignore);
}


//-------------------------------------------------------------------------------------------------
// closest hit 
//

template <typename R, typename P, typename Intersector>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> closest_hit(
        R const& r,
        P begin,
        P end,
        Intersector& isect)
{
    return detail::traverse<false>(r, begin, end, isect);
}


template <typename R, typename P>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> closest_hit(R const& r, P begin, P end)
{
    default_intersector ignore;
    return detail::traverse<false>(r, begin, end, ignore);
}

} // visionaray
