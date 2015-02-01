// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <limits>

#include "macros.h"

namespace visionaray
{

template <typename R, typename P>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> any_hit(R const& r, P begin, P end)
{

    typedef typename R::scalar_type scalar_type;
    typedef P prim_iterator;

    hit_record<R, primitive<unsigned>> result;
    result.hit = false;
    result.t = 3.402823466e+38f;/*std::numeric_limits<scalar_type>::max();*/

    // TODO: any hit..
    for (prim_iterator it = begin; it != end; ++it)
    {
        auto hr = intersect(r, *it);
        if (hr.t > scalar_type(0.0) && hr.t < result.t)
        {
            result.hit |= hr.hit;
            result.t = hr.t;
            result.prim_type = hr.prim_type;
            result.prim_id   = hr.prim_id;
            result.geom_id   = hr.geom_id;
            result.u         = hr.u;
            result.v         = hr.v;

            if (result.hit)
            {
                return result;
            }
        }
    }

    return result;

}

template <typename P>
hit_record<simd::ray4, primitive<unsigned>> any_hit(simd::ray4 const& r, P begin, P end)
{

    typedef simd::float4 scalar_type;
    typedef P prim_iterator;

    hit_record<simd::ray4, primitive<unsigned>> result;
    result.hit = simd::mask4( simd::int4(0, 0, 0, 0) );
    result.t = std::numeric_limits<float>::max();
    result.prim_id = simd::int4(0, 0, 0, 0);

    // TODO: any hit..
    for (prim_iterator it = begin; it != end; ++it)
    {
        auto hr = intersect(r, *it);
        auto closer = hr.hit & ( hr.t >= scalar_type(0.0) && hr.t < result.t );
        result.hit |= closer;
        result.t = select( closer, hr.t, result.t );
        result.prim_type = select( closer, hr.prim_type, result.prim_type );
        result.prim_id   = select( closer, hr.prim_id, result.prim_id );
        result.geom_id   = select( closer, hr.geom_id, result.geom_id );
        result.u         = select( closer, hr.u, result.u );
        result.v         = select( closer, hr.v, result.v );

        if ( all(result.hit) )
        {
            return result;
        }
    }

    return result;

}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
// JESUS...
template <typename P>
hit_record<simd::ray8, primitive<unsigned>> any_hit(simd::ray8 const& r, P begin, P end)
{

    typedef simd::float8 scalar_type;
    typedef P prim_iterator;

    hit_record<simd::ray8, primitive<unsigned>> result;
    result.hit = simd::mask8( simd::int8(0) );
    result.t = std::numeric_limits<float>::max();
    result.prim_id = simd::int8(0);

    // TODO: any hit..
    for (prim_iterator it = begin; it != end; ++it)
    {
        auto hr = intersect(r, *it);
        auto closer = hr.hit & ( hr.t >= scalar_type(0.0) && hr.t < result.t );
        result.hit |= closer;
        result.t = select( closer, hr.t, result.t );
        result.prim_type = select( closer, hr.prim_type, result.prim_type );
        result.prim_id   = select( closer, hr.prim_id, result.prim_id );
        result.geom_id   = select( closer, hr.geom_id, result.geom_id );
        result.u         = select( closer, hr.u, result.u );
        result.v         = select( closer, hr.v, result.v );

        if ( all(result.hit) )
        {
            return result;
        }
    }

    return result;

}
#endif

template <typename R, typename P>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> closest_hit(R const& r, P begin, P end)
{

    typedef typename R::scalar_type scalar_type;
    typedef P prim_iterator;

    hit_record<R, primitive<unsigned>> result;
    result.hit = false;
    result.t = 3.402823466e+38f;//std::numeric_limits<scalar_type>::max();

    for (prim_iterator it = begin; it != end; ++it)
    {
        auto hr = intersect(r, *it);
        if (hr.hit && hr.t >= scalar_type(0.0) && hr.t < result.t)
        {
            result.hit |= hr.hit;
            result.t = hr.t;
            result.prim_type = hr.prim_type;
            result.prim_id   = hr.prim_id;
            result.geom_id   = hr.geom_id;
            result.u         = hr.u;
            result.v         = hr.v;
        }
    }

    return result;

}

template <typename P>
hit_record<simd::ray4, primitive<unsigned>> closest_hit(simd::ray4 const& r, P begin, P end)
{

    typedef simd::float4 scalar_type;
    typedef P prim_iterator;

    hit_record<simd::ray4, primitive<unsigned>> result;
    result.hit = simd::mask4( simd::int4(0, 0, 0, 0) );
    result.t = std::numeric_limits<float>::max();
    result.prim_id = simd::int4(0, 0, 0, 0);

    for (prim_iterator it = begin; it != end; ++it)
    {
        auto hr = intersect(r, *it);
        auto closer = hr.hit & ( hr.t >= scalar_type(0.0) && hr.t < result.t );
        result.hit |= closer;
        result.t = select( closer, hr.t, result.t );
        result.prim_type = select( closer, hr.prim_type, result.prim_type );
        result.prim_id   = select( closer, hr.prim_id, result.prim_id );
        result.geom_id   = select( closer, hr.geom_id, result.geom_id );
        result.u         = select( closer, hr.u, result.u );
        result.v         = select( closer, hr.v, result.v );
    }

    return result;

}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
// JESUS...
template <typename P>
hit_record<simd::ray8, primitive<unsigned>> closest_hit(simd::ray8 const& r, P begin, P end)
{

    typedef simd::float8 scalar_type;
    typedef P prim_iterator;

    hit_record<simd::ray8, primitive<unsigned>> result;
    result.hit = simd::mask8( simd::int8(0) );
    result.t = std::numeric_limits<float>::max();
    result.prim_id = simd::int8(0);

    for (prim_iterator it = begin; it != end; ++it)
    {
        auto hr = intersect(r, *it);
        auto closer = hr.hit & ( hr.t >= scalar_type(0.0) && hr.t < result.t );
        result.hit |= closer;
        result.t = select( closer, hr.t, result.t );
        result.prim_type = select( closer, hr.prim_type, result.prim_type );
        result.prim_id   = select( closer, hr.prim_id, result.prim_id );
        result.geom_id   = select( closer, hr.geom_id, result.geom_id );
        result.u         = select( closer, hr.u, result.u );
        result.v         = select( closer, hr.v, result.v );
    }

    return result;

}
#endif


} // visionaray


