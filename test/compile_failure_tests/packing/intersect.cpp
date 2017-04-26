// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/math.h>
#include <visionaray/array.h>

using namespace visionaray;


int main()
{

#if defined HIT_RECORD_UNPACK_FLOAT4

    hit_record<basic_ray<simd::float4>, primitive<unsigned>> hr;
    array<hit_record<basic_ray<float>, primitive<unsigned>>, 4> hr_array = simd::unpack(hr);

#elif defined HIT_RECORD_UNPACK_FLOAT8

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    hit_record<basic_ray<simd::float8>, primitive<unsigned>> hr;
    array<hit_record<basic_ray<float>, primitive<unsigned>>, 8> hr_array = simd::unpack(hr);
#endif

#elif defined HIT_RECORD_UNPACK_ILLEGAL_LENGTH

    hit_record<basic_ray<simd::float4>, primitive<unsigned>> hr;
    array<hit_record<basic_ray<float>, primitive<unsigned>>, 8> hr_array = simd::unpack(hr);

#elif defined HIT_RECORD_UNPACK_ILLEGAL_INTEGRAL

    hit_record<basic_ray<simd::int4>, primitive<unsigned>> hr;
    auto hr_array = simd::unpack(hr);

#elif defined HIT_RECORD_UNPACK_SINGLE

    hit_record<basic_ray<float>, primitive<unsigned>> hr;
    auto hr_array = simd::unpack(hr);

#endif

    return 0;
}
