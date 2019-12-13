// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SHADE_RECORD_H
#define VSNRAY_SHADE_RECORD_H 1

#include <type_traits>

#include "math/simd/type_traits.h"
#include "math/vector.h"
#include "array.h"

namespace visionaray
{

template <typename T>
struct shade_record
{
    using scalar_type = T;

    vector<3, T> normal;
    vector<3, T> geometric_normal;
    vector<3, T> view_dir;
    vector<3, T> tex_color;
    vector<3, T> light_dir;
    vector<3, T> light_intensity;
};

namespace simd
{

//-------------------------------------------------------------------------------------------------
// Unpack SIMD shade record
//

template <
    typename T,
    typename = typename std::enable_if<is_simd_vector<T>::value>::type,
    typename = typename std::enable_if<
        std::is_floating_point<element_type_t<T>>::value
        >::type
    >
VSNRAY_FUNC
inline array<shade_record<element_type_t<T>>, num_elements<T>::value> unpack(shade_record<T> const& sr)
{
    auto normal           = unpack(sr.normal);
    auto geometric_normal = unpack(sr.geometric_normal);
    auto view_dir         = unpack(sr.view_dir);
    auto tex_color        = unpack(sr.tex_color);
    auto light_dir        = unpack(sr.light_dir);
    auto light_intensity  = unpack(sr.light_intensity);

    array<shade_record<element_type_t<T>>, num_elements<T>::value> result;

    for (int i = 0; i < num_elements<T>::value; ++i)
    {
        result[i].normal           = normal[i];
        result[i].geometric_normal = geometric_normal[i];
        result[i].view_dir         = view_dir[i];
        result[i].tex_color        = tex_color[i];
        result[i].light_dir        = light_dir[i];
        result[i].light_intensity  = light_intensity[i];
    }

    return result;
}

} // simd

} // visionaray

#endif // VSNRAY_SHADE_RECORD_H
