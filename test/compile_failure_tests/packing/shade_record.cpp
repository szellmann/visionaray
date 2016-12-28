// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>

#include <visionaray/math/math.h>
#include <visionaray/shade_record.h>
#include <visionaray/point_light.h>

using namespace visionaray;


int main()
{

#if defined SHADE_RECORD_UNPACK_FLOAT4

    shade_record<point_light<float>, simd::float4> sr;
    std::array<shade_record<point_light<float>, float>, 4> sr_array = simd::unpack(sr);

#elif defined SHADE_RECORD_UNPACK_FLOAT8

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    shade_record<point_light<float>, simd::float8> sr;
    std::array<shade_record<point_light<float>, float>, 8> sr_array = simd::unpack(sr);
#endif

#elif defined SHADE_RECORD_UNPACK_ILLEGAL_LENGTH

    shade_record<point_light<float>, simd::float4> sr;
    std::array<shade_record<point_light<float>, float>, 8> sr_array = simd::unpack(sr);

#elif defined SHADE_RECORD_UNPACK_ILLEGAL_INTEGRAL

    shade_record<point_light<float>, simd::int4> sr;
    auto sr_array = simd::unpack(sr);

#elif defined SHADE_RECORD_UNPACK_SINGLE

    shade_record<point_light<float>, float> sr;
    auto sr_array = simd::unpack(sr);

#endif

    return 0;
}
