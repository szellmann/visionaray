// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/simd/simd.h>
#include <visionaray/math/array.h>
#include <visionaray/math/vector.h>

using namespace visionaray;


const int LEN = VECTOR_LENGTH;


int main()
{

#if defined VECTOR_PACK_FLOAT4

    array<vector<LEN, float>, 4> v_array;
    vector<LEN, simd::float4> v = simd::pack(v_array);

#elif defined VECTOR_PACK_FLOAT8

    array<vector<LEN, float>, 8> v_array;
    vector<LEN, simd::float8> v = simd::pack(v_array);

#elif defined VECTOR_PACK_ILLEGAL_LENGTH_1

    array<vector<LEN, float>, 1> v_array;
    auto v = simd::pack(v_array);

#elif defined VECTOR_PACK_ILLEGAL_LENGTH_3

    array<vector<LEN, float>, 3> v_array;
    auto v = simd::pack(v_array);

#elif defined VECTOR_PACK_ILLEGAL_INTEGRAL

    array<vector<LEN, int>, 4> v_array;
    auto v = simd::pack(v_array);

#elif defined VECTOR_UNPACK_FLOAT4

    vector<LEN, simd::float4> v;
    array<vector<LEN, float>, 4> v_array = simd::unpack(v);

#elif defined VECTOR_UNPACK_FLOAT8

    vector<LEN, simd::float8> v;
    array<vector<LEN, float>, 8> v_array = simd::unpack(v);

#elif defined VECTOR_UNPACK_ILLEGAL_LENGTH

    vector<LEN, simd::float4> v;
    array<vector<LEN, float>, 8> v_array = simd::unpack(v);

#elif defined VECTOR_UNPACK_ILLEGAL_INTEGRAL

    vector<LEN, simd::int4> v;
    auto v_array = simd::unpack(v);

#elif defined VECTOR_UNPACK_SINGLE

    vector<LEN, float> v;
    auto v_array = simd::unpack(v);

#endif

    return 0;
}
