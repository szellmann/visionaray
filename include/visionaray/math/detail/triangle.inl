// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../aabb.h"


namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// Triangle members
//

template <size_t Dim, typename T, typename P>
MATH_FUNC
basic_triangle<Dim, T, P>::basic_triangle(
        vector<Dim, T> const& v1,
        vector<Dim, T> const& e1,
        vector<Dim, T> const& e2
        )
    : v1(v1)
    , e1(e1)
    , e2(e2)
{
}


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
