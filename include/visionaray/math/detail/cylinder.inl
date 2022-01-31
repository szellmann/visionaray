// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../aabb.h"

namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// Cylinder members
//

template <typename T, typename P>
MATH_FUNC
inline basic_cylinder<T, P>::basic_cylinder(
        vector<3, T> const& v1,
        vector<3, T> const& v2,
        T const& r
        )
    : v1(v1)
    , v2(v2)
    , radius(r)
{
}


//-------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T, typename P>
MATH_FUNC
inline basic_aabb<T> get_bounds(basic_cylinder<T, P> const& c)
{
    vector<3, T> r(c.radius);

    basic_aabb<T> result;
    result.invalidate();
    result.insert(c.v1 - r);
    result.insert(c.v1 + r);
    result.insert(c.v2 - r);
    result.insert(c.v2 + r);
    return result;
}

} // MATH_NAMESPACE
