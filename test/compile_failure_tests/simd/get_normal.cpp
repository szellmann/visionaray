// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/get_normal.h>
#include <visionaray/get_shading_normal.h>

using namespace visionaray;


int main()
{

#if defined GET_NORMAL_TRIANGLE_PERFACE_FLOAT4

    vector<3, float>* normals;
    hit_record<basic_ray<simd::float4>, primitive<unsigned>> hr;
    vector<3, simd::float4> n = get_normal(normals, hr, basic_triangle<3, float, unsigned>{}, normals_per_face_binding{});

#elif defined GET_NORMAL_TRIANGLE_PERFACE_FLOAT8

    vector<3, float>* normals;
    hit_record<basic_ray<simd::float8>, primitive<unsigned>> hr;
    vector<3, simd::float8> n = get_normal(normals, hr, basic_triangle<3, float, unsigned>{}, normals_per_face_binding{});

#elif defined GET_NORMAL_TRIANGLE_PERFACE_INT4

    vector<3, float>* normals;
    hit_record<basic_ray<simd::int4>, primitive<unsigned>> hr;
    auto n = get_normal(normals, hr, basic_triangle<3, float, unsigned>{}, normals_per_face_binding{});

#elif defined GET_NORMAL_TRIANGLE_PERFACE_MASK4

    vector<3, float>* normals;
    hit_record<basic_ray<simd::mask4>, primitive<unsigned>> hr;
    auto n = get_normal(normals, hr, basic_triangle<3, float, unsigned>{}, normals_per_face_binding{});

#elif defined GET_NORMAL_TRIANGLE_PERVERTEX_FLOAT4

    vector<3, float>* normals;
    hit_record<basic_ray<simd::float4>, primitive<unsigned>> hr;
    vector<3, simd::float4> n = get_shading_normal(normals, hr, basic_triangle<3, float, unsigned>{}, normals_per_vertex_binding{});

#elif defined GET_NORMAL_TRIANGLE_PERVERTEX_FLOAT8

    vector<3, float>* normals;
    hit_record<basic_ray<simd::float8>, primitive<unsigned>> hr;
    vector<3, simd::float8> n = get_shading_normal(normals, hr, basic_triangle<3, float, unsigned>{}, normals_per_vertex_binding{});

#elif defined GET_NORMAL_TRIANGLE_PERVERTEX_INT4

    vector<3, float>* normals;
    hit_record<basic_ray<simd::int4>, primitive<unsigned>> hr;
    auto n = get_shading_normal(normals, hr, basic_triangle<3, float, unsigned>{}, normals_per_vertex_binding{});

#elif defined GET_NORMAL_TRIANGLE_PERVERTEX_MASK4

    vector<3, float>* normals;
    hit_record<basic_ray<simd::mask4>, primitive<unsigned>> hr;
    auto n = get_shading_normal(normals, hr, basic_triangle<3, float, unsigned>{}, normals_per_vertex_binding{});

#endif

    return 0;
}
