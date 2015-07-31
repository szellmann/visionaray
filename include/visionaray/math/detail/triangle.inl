// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../aabb.h"


namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// Geometric functions
//

template <size_t Dim, typename T, typename P>
MATH_FUNC
basic_aabb<T> get_bounds(basic_triangle<Dim, T, P> const& t)
{
    basic_aabb<T> bounds;

    bounds.invalidate();
    bounds.insert(t.v1);
    bounds.insert(t.v1 + t.e1);
    bounds.insert(t.v1 + t.e2);

    return bounds;
}

} // MATH_NAMESPACE
