// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/math.h>
#include <visionaray/shade_record.h>

using namespace visionaray;


int main()
{

#if defined SHADE_RECORD_UNPACK_FLOAT4

    shade_record<simd::float4> sr;
    array<shade_record<float>, 4> sr_array = simd::unpack(sr);

#elif defined SHADE_RECORD_UNPACK_FLOAT8

    shade_record<simd::float8> sr;
    array<shade_record<float>, 8> sr_array = simd::unpack(sr);

#elif defined SHADE_RECORD_UNPACK_ILLEGAL_LENGTH

    shade_record<simd::float4> sr;
    array<shade_record<float>, 8> sr_array = simd::unpack(sr);

#elif defined SHADE_RECORD_UNPACK_ILLEGAL_INTEGRAL

    shade_record<simd::int4> sr;
    auto sr_array = simd::unpack(sr);

#elif defined SHADE_RECORD_UNPACK_SINGLE

    shade_record<float> sr;
    auto sr_array = simd::unpack(sr);

#endif

    return 0;
}
