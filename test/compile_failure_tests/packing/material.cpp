// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/array.h>
#include <visionaray/material.h>

using namespace visionaray;


// Material type to check
#if defined MATERIAL_MATTE
    template <typename T>
    using mat_type = matte<T>;
#elif defined MATERIAL_EMISSIVE
    template <typename T>
    using mat_type = emissive<T>;
#elif defined MATERIAL_PLASTIC
    template <typename T>
    using mat_type = plastic<T>;
#elif defined MATERIAL_MIRROR
    template <typename T>
    using mat_type = mirror<T>;
#endif


int main()
{

#if defined MATERIAL_PACK_FLOAT4

    array<mat_type<float>, 4> mat_array;
    mat_type<simd::float4> mat = simd::pack(mat_array);

#elif defined MATERIAL_PACK_FLOAT8

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    array<mat_type<float>, 8> mat_array;
    mat_type<simd::float8> mat = simd::pack(mat_array);
#endif

#elif defined MATERIAL_PACK_ILLEGAL_LENGTH_1

    array<mat_type<float>, 1> mat_array;
    auto mat = simd::pack(mat_array);

#elif defined MATERIAL_PACK_ILLEGAL_LENGTH_3

    array<mat_type<float>, 3> mat_array;
    auto mat = simd::pack(mat_array);

#elif defined MATERIAL_PACK_ILLEGAL_INTEGRAL

    array<mat_type<int>, 4> mat_array;
    auto mat = simd::pack(mat_array);

#endif

    return 0;
}
