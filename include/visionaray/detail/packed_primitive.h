// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_PACKED_PRIMITIVE_H
#define VSNRAY_DETAIL_PACKED_PRIMITIVE_H 1

#include "../math/simd/simd.h"
#include "../math/triangle.h"

namespace visionaray::detail
{

//-------------------------------------------------------------------------------------------------
// Traits mapping scalar to packed type (default: no packing)
//

template <typename P, int W>
struct packed_primitive
{
    using type = P;
};

template <>
struct packed_primitive<basic_triangle<3, float>, 4>
{
    using type = basic_triangle<3, simd::float4, simd::int4>;
};

template <>
struct packed_primitive<basic_triangle<3, float>, 8>
{
    using type = basic_triangle<3, simd::float8, simd::int8>;
};

} // visionaray::detail

#endif // VSNRAY_DETAIL_PACKED_PRIMITIVE_H
