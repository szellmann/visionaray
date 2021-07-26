// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_PIXEL_SAMPLER_TYPES_H
#define VSNRAY_PIXEL_SAMPLER_TYPES_H 1

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

// Uniform (change spp to 2x, 4x, or 8x for SSAA)
struct uniform_type : base_type
{
    unsigned ssaa_factor = 1; // 1,2,4,8
};

// Jittered pixel positions
struct jittered_type : base_type {};

// Jittered and successive blending
template <typename T>
struct basic_jittered_blend_type : jittered_type
{
    unsigned spp = 1;

    T sfactor;
    T dfactor;
};

using jittered_blend_type = basic_jittered_blend_type<float>;

} // pixel_sampler
} // visionaray

#endif // VSNRAY_PIXEL_SAMPLER_TYPES_H
