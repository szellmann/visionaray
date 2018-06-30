// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RESULT_RECORD_H
#define VSNRAY_RESULT_RECORD_H 1

#include "math/simd/type_traits.h"
#include "math/vector.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Result record that the builtin visionaray kernels return
//

template <typename T>
struct result_record
{
    using scalar_type = T;
    using mask_type   = simd::mask_type_t<T>;
    using vec_type    = vector<3, T>;
    using color_type  = vector<4, T>;

    mask_type   hit       = false;
    color_type  color     = color_type(0.0);
    scalar_type depth     = scalar_type(0.0);
    vec_type    isect_pos = vec_type(0.0);
};

} // visionaray

#endif // VSNRAY_RESULT_RECORD_H
