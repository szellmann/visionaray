// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../constants.h"
#include "math.h"


namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// Sphere members
//

template <typename T, typename P>
MATH_FUNC
inline basic_sphere<T, P>::basic_sphere(vector<3, T> const& c, T r)
    : center(c)
    , radius(r)
{
}


//-------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T, typename P>
MATH_FUNC
inline basic_aabb<T> get_bounds(basic_sphere<T, P> const& s)
{
    basic_aabb<T> bounds;

    bounds.invalidate();
    bounds.insert(s.center - s.radius);
    bounds.insert(s.center + s.radius);

    return bounds;
}


template <typename T, typename P>
MATH_FUNC
inline T volume(basic_sphere<T, P> const& s)
{
    auto r3 = s.radius * s.radius * s.radius;
    return T(4.0) / T(3.0) * constants::pi<T>() * r3;
}

} // MATH_NAMESPACE
