// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_FILTER_ARITHMETIC_TYPES_H
#define VSNRAY_TEXTURE_DETAIL_FILTER_ARITHMETIC_TYPES_H 1

#include <cstddef>

#include <visionaray/math/vector.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Deduce types used by the filtering algorithm based on the texture and coordinate types
//

// Scalar texture, we can use float for internal arithmetic
template <typename TexelType, typename CoordinateType>
struct arithmetic_types
{
    // Type used for internal calculations by the filter functions
    using internal_type = float;

    // Type returned by the filter functions
    using return_type = TexelType;

    // Integer type, for example for texture sizes
    using int_type = unsigned;
};

// Vector texture, we can use float for internal arithmetic
template <size_t Dim, typename T, typename CoordinateType>
struct arithmetic_types<vector<Dim, T>, CoordinateType>
{
    // Type used for internal calculations by the filter functions
    using internal_type = vector<Dim, float>;

    // Type returned by the filter functions
    using return_type = vector<Dim, T>;

    // Integer type, for example for texture sizes
    using int_type = unsigned;
};

} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_FILTER_ARITHMETIC_TYPES_H
