// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MAKE_RANDOM_SEED_H
#define VSNRAY_MAKE_RANDOM_SEED_H 1

#include "detail/macros.h"
#include "math/simd/type_traits.h"
#include "array.h"
#include "packet_traits.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Make random seed for LCG
// Taken from https://github.com/owl-project/owl/
//

VSNRAY_FUNC
inline unsigned make_random_seed(int val0, int val1)
{
    unsigned v0 = val0;
    unsigned v1 = val1;
    unsigned s0 = 0;

    for (unsigned n = 0; n < 4; ++n)
    {
      s0 += 0x9E3779B9u;
      v0 += ((v1 << 4) + 0xA341316Cu) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4u);
      v1 += ((v0 << 4) + 0xAd90777Du) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761Eu);
    }

    return v0;
}


//-------------------------------------------------------------------------------------------------
// Overload for SIMD vectors
//

template <typename I>
VSNRAY_FUNC
inline array<unsigned, simd::num_elements<I>::value> make_random_seed(I val0, I val1)
{
    using S = simd::float_type_t<I>;
    using IA = simd::aligned_array_t<I>;

    array<unsigned, simd::num_elements<I>::value> result;
    IA arr_v0;
    IA arr_v1;

    simd::store(arr_v0, val0);
    simd::store(arr_v1, val1);

    int pw = packet_size<S>::w;
    int ph = packet_size<S>::h;

    for (int i = 0; i < pw * ph; ++i)
    {
        result[i] = make_random_seed(arr_v0[i], arr_v1[i]);
    }

    return result;
}

} // visionaray

#endif // VSNRAY_MAKE_RANDOM_SEED_H
