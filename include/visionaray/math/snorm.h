// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_SNORM_H
#define VSNRAY_MATH_SNORM_H 1

#include <cstdint>

#include "config.h"

namespace MATH_NAMESPACE
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Best signed integer type from bit depth. TODO: support arbitrary bit depth
//

template <unsigned Bits> struct best_int { /* No best_int::type causes a compiler error */ };
template <> struct best_int< 8> { typedef int8_t  type; };
template <> struct best_int<16> { typedef int16_t type; };
template <> struct best_int<32> { typedef int32_t type; };

} // detail


//-------------------------------------------------------------------------------------------------
// snorm
//

template <unsigned Bits>
class snorm
{
public:

    using value_type = typename detail::best_int<Bits>::type;

public:

    value_type value;

    MATH_FUNC snorm() = default;

    MATH_FUNC /* implicit */ snorm(float f);

    MATH_FUNC operator value_type() const;
    MATH_FUNC operator float() const;
};

} // MATH_NAMESPACE

#include "detail/snorm.inl"

#endif // VSNRAY_MATH_SNORM_H
