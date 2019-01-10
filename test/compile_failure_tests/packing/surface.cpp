// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/array.h>
#include <visionaray/math/forward.h>
#include <visionaray/material.h>
#include <visionaray/get_surface.h>

using namespace visionaray;


// Material type to check
template <typename T>
using mat_type = matte<T>;


template <typename N, typename M, int SIZE>
using param_type = array<surface<N, vec3f, M>, SIZE>;
template <typename N, typename M>
using result_type = surface<N, N/*TODO: in general TexCol != N*/, M>;


int main()
{

#if defined SURFACE_PACK_FLOAT4

    param_type<vec3f, mat_type<float>, 4> surf_array;
    result_type<vector<3, simd::float4>, mat_type<simd::float4>> surf = simd::pack(surf_array);

#elif defined SURFACE_PACK_FLOAT8

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    param_type<vec3f, mat_type<float>, 8> surf_array;
    result_type<vector<3, simd::float8>, mat_type<simd::float8>> surf = simd::pack(surf_array);
#endif

#elif defined SURFACE_PACK_ILLEGAL_LENGTH_1

    param_type<vec3f, mat_type<float>, 1> surf_array;
    auto surf = simd::pack(surf_array);

#elif defined SURFACE_PACK_ILLEGAL_LENGTH_3

    param_type<vec3f, mat_type<float>, 3> surf_array;
    auto surf = simd::pack(surf_array);

#elif defined SURFACE_PACK_ILLEGAL_INTEGRAL_1

    param_type<vec3f, mat_type<int>, 4> surf_array;
    auto surf = simd::pack(surf_array);

#elif defined SURFACE_PACK_ILLEGAL_INTEGRAL_2

    param_type<vec3i, mat_type<float>, 4> surf_array;
    auto surf = simd::pack(surf_array);

#endif

    return 0;
}
