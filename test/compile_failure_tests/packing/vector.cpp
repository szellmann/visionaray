// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>

#include <visionaray/math/vector.h>

using namespace visionaray;


const int LEN = VECTOR_LENGTH;


int main()
{

#if defined VECTOR_PACK_FLOAT4

    std::array<vector<LEN, float>, 4> v_array;
    vector<LEN, simd::float4> v = simd::pack(v_array);

#elif defined VECTOR_PACK_FLOAT8

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
    std::array<vector<LEN, float>, 8> v_array;
    vector<LEN, simd::float8> v = simd::pack(v_array);
#endif

#elif defined VECTOR_PACK_ILLEGAL_LENGTH_1

    std::array<vector<LEN, float>, 1> v_array;
    auto v = simd::pack(v_array);

#elif defined VECTOR_PACK_ILLEGAL_LENGTH_3

    std::array<vector<LEN, float>, 3> v_array;
    auto v = simd::pack(v_array);

#elif defined VECTOR_PACK_ILLEGAL_INTEGRAL

    std::array<vector<LEN, int>, 4> v_array;
    auto v = simd::pack(v_array);

#elif defined VECTOR_UNPACK_FLOAT4

    vector<LEN, simd::float4> v;
    std::array<vector<LEN, float>, 4> v_array = simd::unpack(v);

#elif defined VECTOR_UNPACK_FLOAT8

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
    vector<LEN, simd::float8> v;
    std::array<vector<LEN, float>, 8> v_array = simd::unpack(v);
#endif

#elif defined VECTOR_UNPACK_ILLEGAL_LENGTH

    vector<LEN, simd::float4> v;
    std::array<vector<LEN, float>, 8> v_array = simd::unpack(v);

#elif defined VECTOR_UNPACK_ILLEGAL_INTEGRAL

    vector<LEN, simd::int4> v;
    auto v_array = simd::unpack(v);

#elif defined VECTOR_UNPACK_SINGLE

    vector<LEN, float> v;
    auto v_array = simd::unpack(v);

#endif

    return 0;
}
