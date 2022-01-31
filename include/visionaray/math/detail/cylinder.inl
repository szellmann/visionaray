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
    vector<3, T> a = c.v2 - c.v1;
    T aa = dot(a, a);

    vector<3, T> e(
        c.radius * sqrt(T(1.0) - a.x * a.x / aa),
        c.radius * sqrt(T(1.0) - a.y * a.y / aa),
        c.radius * sqrt(T(1.0) - a.z * a.z / aa)
        );

    vector<3, T> pa = min(c.v1 - e, c.v2 - e);
    vector<3, T> pb = max(c.v1 - e, c.v2 - e);

    basic_aabb<T> result;
    result.invalidate();
    result.insert(pa);
    result.insert(pb);
    return result;
}

} // MATH_NAMESPACE
