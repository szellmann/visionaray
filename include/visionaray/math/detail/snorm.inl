// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <type_traits>

#include "../simd/type_traits.h"
#include "math.h"

namespace MATH_NAMESPACE
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Convert float to snorm (cf. OpenGL 4.4, 2.3.4.1)
//

template <unsigned Bits>
MATH_FUNC
inline int32_t float_to_snorm(float f)
{
    f = clamp(f, -1.0f, 1.0f);
    return static_cast<int32_t>(f * static_cast<double>((1ULL << (Bits - 1)) - 1));
}

template <unsigned Bits>
MATH_FUNC
inline float snorm_to_float(int32_t n)
{
    return max( static_cast<float>(n) / static_cast<double>((1ULL << (Bits - 1)) - 1), -1.0 );
}

} // detail


//-------------------------------------------------------------------------------------------------
// snorm members
//

template <unsigned Bits>
MATH_FUNC
inline snorm<Bits>::snorm(float f)
    : value(detail::float_to_snorm<Bits>(f))
{
}

template <unsigned Bits>
MATH_FUNC
inline snorm<Bits>::operator value_type() const
{
    return value;
}

template <unsigned Bits>
MATH_FUNC
inline snorm<Bits>::operator float() const
{
    return detail::snorm_to_float<Bits>(value);
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

template <unsigned Bits>
MATH_FUNC
inline bool operator==(snorm<Bits> a, snorm<Bits> b)
{
    using T = typename snorm<Bits>::value_type;

    return static_cast<T>(a) == static_cast<T>(b);
}

template <unsigned Bits>
MATH_FUNC
inline bool operator!=(snorm<Bits> a, snorm<Bits> b)
{
    using T = typename snorm<Bits>::value_type;

    return static_cast<T>(a) != static_cast<T>(b);
}

template <unsigned Bits>
MATH_FUNC
inline bool operator<(snorm<Bits> a, snorm<Bits> b)
{
    using T = typename snorm<Bits>::value_type;

    return static_cast<T>(a) < static_cast<T>(b);
}

template <unsigned Bits>
MATH_FUNC
inline bool operator<=(snorm<Bits> a, snorm<Bits> b)
{
    using T = typename snorm<Bits>::value_type;

    return static_cast<T>(a) <= static_cast<T>(b);
}

template <unsigned Bits>
MATH_FUNC
inline bool operator>(snorm<Bits> a, snorm<Bits> b)
{
    using T = typename snorm<Bits>::value_type;

    return static_cast<T>(a) > static_cast<T>(b);
}

template <unsigned Bits>
MATH_FUNC
inline bool operator>=(snorm<Bits> a, snorm<Bits> b)
{
    using T = typename snorm<Bits>::value_type;

    return static_cast<T>(a) >= static_cast<T>(b);
}

} // MATH_NAMESPACE
