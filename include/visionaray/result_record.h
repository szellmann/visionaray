// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RESULT_RECORD_H
#define VSNRAY_RESULT_RECORD_H 1

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
    using mask_type   = simd::mask_type_t<T>;
    using vec_type    = vector<3, T>;
    using color_type  = vector<4, T>;

public:

    VSNRAY_FUNC result_record()
        : hit(false)
        , color(0.0)
        , depth(0.0)
        , isect_pos(0.0)
    {
    }

    mask_type   hit;
    color_type  color;
    scalar_type depth;
    vec_type    isect_pos;

};

} // visionaray

#endif // VSNRAY_RESULT_RECORD_H
