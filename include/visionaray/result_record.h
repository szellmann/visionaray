// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RESULT_RECORD_H
#define VSNRAY_RESULT_RECORD_H

#include "math/math.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Result record that the builtin visionaray kernels return
//

template <typename T>
class result_record
{
public:

    using scalar_type = T;
    using vec_type    = vector<3, T>;
    using color_type  = vector<4, T>;

public:

    bool        hit = false;
    color_type  color;
    scalar_type depth;
    vec_type    isect_pos;

};

template <>
class result_record<simd::float4>
{
public:

    using scalar_type = simd::float4;
    using vec_type    = vector<3, simd::float4>;
    using color_type  = vector<4, simd::float4>;

public:

    simd::mask4 hit = false;
    color_type  color;
    scalar_type depth;
    vec_type    isect_pos;

};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <>
class result_record<simd::float8>
{
public:

    using scalar_type = simd::float4;
    using vec_type    = vector<3, simd::float8>;
    using color_type  = vector<4, simd::float8>;

public:

    simd::mask8 hit = false;
    color_type  color;
    scalar_type depth;
    vec_type    isect_pos;

};

#endif

} // visionaray

#endif // VSNRAY_RESULT_RECORD_H
