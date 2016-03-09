// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>

#include "visionaray/math/forward.h"
#include "visionaray/material.h"
#include "visionaray/get_surface.h"

using namespace visionaray;


// Material type to check
template <typename T>
using mat_type = matte<T>;


#if defined WITHOUT_TEX_COLOR
    template <typename N, typename M, int SIZE>
    using param_type = std::array<surface<N, M>, SIZE>;
    template <typename N, typename M>
    using result_type = surface<N, M>;
#elif defined WITH_TEX_COLOR
    template <typename N, typename M, int SIZE>
    using param_type = std::array<surface<N, M, vec3f>, SIZE>;
    template <typename N, typename M>
    using result_type = surface<N, M, N>;
#endif


int main()
{

#if defined SURFACE_PACK_FLOAT4

    param_type<vec3f, mat_type<float>, 4> surf_array;
    result_type<vector<3, simd::float4>, mat_type<simd::float4>> surf = simd::pack(surf_array);

#elif defined SURFACE_PACK_FLOAT8

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
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
