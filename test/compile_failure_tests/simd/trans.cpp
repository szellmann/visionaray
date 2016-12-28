// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/math.h>

using namespace visionaray;


#define TEST_FUNC simd::detail::pow2
//#define TEST_FUNC simd::log
//#define TEST_FUNC simd::exp
//#define TEST_FUNC simd::log2


int main()
{

#if defined TRANS_FLOAT4

    simd::float4 f;
    simd::float4 g = TEST_FUNC(f);

#elif defined TRANS_FLOAT8

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    simd::float8 f;
    simd::float8 g = TEST_FUNC(f);
#endif

#elif defined TRANS_FLOAT

    float f;
    auto g = TEST_FUNC(f);

#elif defined TRANS_INT4

    simd::int4 f;
    auto g = TEST_FUNC(f);

#elif defined TRANS_MASK4

    simd::mask4 f;
    auto g = TEST_FUNC(f);

#endif

    return 0;
}
