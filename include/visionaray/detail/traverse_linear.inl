// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "macros.h"

namespace visionaray
{
namespace detail
{

template <bool AnyHit, typename R, typename P>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> traverse(R const& r, P begin, P end, typename R::scalar_type max_t)
{

    typedef typename R::scalar_type scalar_type;
    typedef P prim_iterator;

    hit_record<R, primitive<unsigned>> result;
    result.hit = false;
    result.t = numeric_limits<float>::max();
    result.prim_id = 0;

    for (prim_iterator it = begin; it != end; ++it)
    {
        auto hr = intersect(r, *it);
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

template <bool AnyHit, typename R, typename P>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> traverse(R const& r, P begin, P end)
{
    return traverse<AnyHit>(r, begin, end, numeric_limits<float>::max());
}

} // detail


template <typename R, typename P>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> any_hit(R const& r, P begin, P end)
{
    return detail::traverse<true>(r, begin, end);
}

template <typename R, typename P>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> any_hit(R const& r, P begin, P end, typename R::scalar_type max_t)
{
    return detail::traverse<true>(r, begin, end, max_t);
}

template <typename R, typename P>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> closest_hit(R const& r, P begin, P end)
{
    return detail::traverse<false>(r, begin, end);
}


} // visionaray
