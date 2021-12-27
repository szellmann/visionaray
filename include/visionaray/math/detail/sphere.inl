// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/sampling.h>

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
inline T area(basic_sphere<T, P> const& s)
{
    return T(4.0) * constants::pi<T>() * s.radius * s.radius;
}

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

template <typename T, typename P, typename Generator, typename U = typename Generator::value_type>
MATH_FUNC
inline vector<3, U> sample_surface(basic_sphere<T, P> const& s, vector<3, U> const& reference_point,  Generator& gen)
{
    VSNRAY_UNUSED(reference_point);
    return uniform_sample_sphere(gen.next(), gen.next()) * U(s.radius) + vector<3, U>(s.center);
}

template <typename T, typename P>
MATH_FUNC
inline T volume(basic_sphere<T, P> const& s)
{
    auto r3 = s.radius * s.radius * s.radius;
    return T(4.0) / T(3.0) * constants::pi<T>() * r3;
}

} // MATH_NAMESPACE
