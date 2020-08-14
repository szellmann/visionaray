// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_PIXEL_SAMPLER_TYPES_H
#define VSNRAY_PIXEL_SAMPLER_TYPES_H 1

#include <cstddef>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Pixel sampler types for use in scheduler params
//

namespace pixel_sampler
{

// Pixel sampler base -------------------------------------

struct base_type {};


// Built-in pixel sampler types ---------------------------

// Supersampling anti-aliasing
template <size_t NumSamples>
struct ssaa_type : base_type {};

// 1x SSAA (no supersampling)
using uniform_type = ssaa_type<1>;

// 1x SSAA and successive blending
template <typename T>
struct basic_uniform_blend_type : uniform_type
{
    T sfactor;
    T dfactor;
};

using uniform_blend_type = basic_uniform_blend_type<float>;

// Jittered pixel positions
struct jittered_type : base_type {};

// Jittered and successive blending
template <typename T>
struct basic_jittered_blend_type : jittered_type
{
    T sfactor;
    T dfactor;
};

using jittered_blend_type = basic_jittered_blend_type<float>;

} // pixel_sampler
} // visionaray

#endif // VSNRAY_PIXEL_SAMPLER_TYPES_H
