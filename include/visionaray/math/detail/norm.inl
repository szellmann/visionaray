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


//-------------------------------------------------------------------------------------------------
// Convert float to unorm (cf. OpenGL 4.4, 2.3.4.1)
//

template <unsigned Bits>
MATH_FUNC
inline uint32_t float_to_unorm(float f)
{
    f = saturate(f);
    return static_cast<uint32_t>(f * static_cast<double>((1ULL << Bits) - 1));
}

template <unsigned Bits>
MATH_FUNC
inline float unorm_to_float(uint32_t u)
{
    return static_cast<float>(u) / static_cast<double>((1ULL << Bits) - 1);
}


//-------------------------------------------------------------------------------------------------
// Some special overloads for SIMD types
// So far only valid for 8-bit and 16-bit because the double trick won't work here!
//

template <
    unsigned Bits,
    typename I,
    typename = typename std::enable_if<simd::is_simd_vector<I>::value>::type
    >
MATH_FUNC
inline typename simd::float_type<I>::type unorm_to_float(I const& u)
{
    using F = typename simd::float_type<I>::type;
    return F(u) / F(static_cast<float>((1ULL << Bits) - 1));
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


//-------------------------------------------------------------------------------------------------
// unorm members
//

template <unsigned Bits>
MATH_FUNC
inline unorm<Bits>::unorm(float f)
    : value(detail::float_to_unorm<Bits>(f))
{
}

template <unsigned Bits>
MATH_FUNC
inline unorm<Bits>::operator value_type() const
{
    return value;
}

template <unsigned Bits>
MATH_FUNC
inline unorm<Bits>::operator float() const
{
    return detail::unorm_to_float<Bits>(value);
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

template <unsigned Bits>
MATH_FUNC
inline bool operator==(unorm<Bits> a, unorm<Bits> b)
{
    using T = typename unorm<Bits>::value_type;

    return static_cast<T>(a) == static_cast<T>(b);
}

template <unsigned Bits>
MATH_FUNC
inline bool operator!=(unorm<Bits> a, unorm<Bits> b)
{
    using T = typename unorm<Bits>::value_type;

    return static_cast<T>(a) != static_cast<T>(b);
}

template <unsigned Bits>
MATH_FUNC
inline bool operator<(unorm<Bits> a, unorm<Bits> b)
{
    using T = typename unorm<Bits>::value_type;

    return static_cast<T>(a) < static_cast<T>(b);
}

template <unsigned Bits>
MATH_FUNC
inline bool operator<=(unorm<Bits> a, unorm<Bits> b)
{
    using T = typename unorm<Bits>::value_type;

    return static_cast<T>(a) <= static_cast<T>(b);
}

template <unsigned Bits>
MATH_FUNC
inline bool operator>(unorm<Bits> a, unorm<Bits> b)
{
    using T = typename unorm<Bits>::value_type;

    return static_cast<T>(a) > static_cast<T>(b);
}

template <unsigned Bits>
MATH_FUNC
inline bool operator>=(unorm<Bits> a, unorm<Bits> b)
{
    using T = typename unorm<Bits>::value_type;

    return static_cast<T>(a) >= static_cast<T>(b);
}

} // MATH_NAMESPACE
