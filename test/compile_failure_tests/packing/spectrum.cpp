// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>

#include <visionaray/spectrum.h>

using namespace visionaray;

int main()
{

#if defined SPECTRUM_PACK_FLOAT4

    std::array<spectrum<float>, 4> spec_array;
    spectrum<simd::float4> spec = simd::pack(spec_array);

#elif defined SPECTRUM_PACK_FLOAT8

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    std::array<spectrum<float>, 8> spec_array;
    spectrum<simd::float8> spec = simd::pack(spec_array);
#endif

#elif defined SPECTRUM_PACK_ILLEGAL_LENGTH_1

    std::array<spectrum<float>, 1> spec_array;
    auto spec = simd::pack(spec_array);

#elif defined SPECTRUM_PACK_ILLEGAL_LENGTH_3

    std::array<spectrum<float>, 3> spec_array;
    auto spec = simd::pack(spec_array);

#elif defined SPECTRUM_PACK_ILLEGAL_INTEGRAL

    std::array<spectrum<int>, 4> spec_array;
    auto spec = simd::pack(spec_array);

#endif

    return 0;
}
