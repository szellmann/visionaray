// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_LIGHT_SAMPLE_H
#define VSNRAY_LIGHT_SAMPLE_H 1

#include "math/simd/type_traits.h"
#include "math/vector.h"

namespace visionaray
{

template <typename T>
struct light_sample
{
    // Sampled direction
    vector<3, T> dir;

    // Distance to sample
    T dist;

    // Light intensity at pos
    vector<3, T> intensity;

    // Light normal at pos
    vector<3, T> normal;

    // Area of the sampled light source
    T area;

    // Indicates if sample was generated from a delta light
    simd::mask_type_t<T> delta_light;
};

} // visionaray

#endif // VSNRAY_LIGHT_SAMPLE_H
