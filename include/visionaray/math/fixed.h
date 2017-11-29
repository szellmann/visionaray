// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_FIXED_H
#define VSNRAY_MATH_FIXED_H 1

#include <cstdint>

#include "config.h"

namespace visionaray
{

template <unsigned Bits>
struct best_fixed_rep;

template <>
struct best_fixed_rep<8>
{
    using type = int8_t;
};

template <>
struct best_fixed_rep<16>
{
    using type = int16_t;
};

template <>
struct best_fixed_rep<32>
{
    using type = int32_t;
};

template <>
struct best_fixed_rep<64>
{
    using type = int64_t;
};


//-------------------------------------------------------------------------------------------------
// signed fixed-point type (I = sign bit + integer bits, F = fractional bits)
//
// The implementation of this type relies on the compiler performing arithmetic right shifts!
//

template <unsigned I, unsigned F>
class fixed
{
public:

    using rep_type = typename best_fixed_rep<I + F>::type;

public:

    fixed() = default;

    MATH_FUNC /* implicit */ fixed(char c);
    MATH_FUNC /* implicit */ fixed(short s);
    MATH_FUNC /* implicit */ fixed(int i);
    MATH_FUNC /* implicit */ fixed(long l);
    MATH_FUNC /* implicit */ fixed(long long ll);

    MATH_FUNC /* implicit */ fixed(unsigned char uc);
    MATH_FUNC /* implicit */ fixed(unsigned short us);
    MATH_FUNC /* implicit */ fixed(unsigned int ui);
    MATH_FUNC /* implicit */ fixed(unsigned long ul);
    MATH_FUNC /* implicit */ fixed(unsigned long long ull);

    MATH_FUNC /* implicit */ fixed(float f);
    MATH_FUNC /* implicit */ fixed(double d);
    MATH_FUNC /* implicit */ fixed(long double ld);

    MATH_FUNC /* implicit */ operator char() const;
    MATH_FUNC /* implicit */ operator short() const;
    MATH_FUNC /* implicit */ operator int() const;
    MATH_FUNC /* implicit */ operator long() const;
    MATH_FUNC /* implicit */ operator long long() const;

    MATH_FUNC /* implicit */ operator unsigned char() const;
    MATH_FUNC /* implicit */ operator unsigned short() const;
    MATH_FUNC /* implicit */ operator unsigned int() const;
    MATH_FUNC /* implicit */ operator unsigned long() const;
    MATH_FUNC /* implicit */ operator unsigned long long() const;

    MATH_FUNC /* implicit */ operator float() const;
    MATH_FUNC /* implicit */ operator double() const;
    MATH_FUNC /* implicit */ operator long double() const;

private:

    rep_type rep_;

};

} // visionaray

#include "detail/fixed.inl"

#endif // VSNRAY_MATH_FIXED_H
