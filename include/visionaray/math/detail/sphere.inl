// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../detail/math.h"


namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T>
MATH_FUNC
T volume(basic_sphere<T> const& s)
{
    auto r3 = s.radius * s.radius * s.radius;
    return T(4.0) / T(3.0) * constants::pi<T>() * r3;
}

} // MATH_NAMESPACE
