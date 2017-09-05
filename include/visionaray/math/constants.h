// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_CONSTANTS_H
#define VSNRAY_MATH_CONSTANTS_H 1

#include "config.h"

namespace MATH_NAMESPACE
{
namespace constants
{

//--------------------------------------------------------------------------------------------------
// Constants
//

template <typename T> MATH_FUNC T degrees_to_radians()  { return T(1.74532925199432957692369076849e-02); }
template <typename T> MATH_FUNC T radians_to_degrees()  { return T(5.72957795130823208767981548141e+01); }
template <typename T> MATH_FUNC T e()                   { return T(2.71828182845904523536028747135e+00); }
template <typename T> MATH_FUNC T log2_e()              { return T(1.44269504088896338700465094007e+00); }
template <typename T> MATH_FUNC T pi()                  { return T(3.14159265358979323846264338328e+00); }
template <typename T> MATH_FUNC T two_pi()              { return T(6.28318530717958623199592693709e+00); }
template <typename T> MATH_FUNC T inv_pi()              { return T(3.18309886183790691216444201928e-01); }
template <typename T> MATH_FUNC T pi_over_two()         { return T(1.57079632679489655799898173427e+00); }
template <typename T> MATH_FUNC T pi_over_four()        { return T(7.85398163397448278999490867136e-01); }

} // constants
} // MATH_NAMESPACE

#endif // VSNRAY_MATH_CONSTANTS_H
