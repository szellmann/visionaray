#include "visionaray/get_color.h"
#include "visionaray/math/primitive.h"

using namespace visionaray;


int main()
{

#if defined GET_COLOR_FLOAT4

    vector<3, float> *colors;
    hit_record<basic_ray<simd::float4>, primitive<unsigned>> hr;
    vector<3, simd::float4> c = get_color(colors, hr, primitive<unsigned>{}, per_face_binding{});

#elif defined GET_COLOR_FLOAT8

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
    vector<3, float> *colors;
    hit_record<basic_ray<simd::float8>, primitive<unsigned>> hr;
    vector<3, simd::float8> c = get_color(colors, hr, primitive<unsigned>{}, per_face_binding{});
#endif

#elif defined GET_COLOR_INT4

    vector<3, float> *colors;
    hit_record<basic_ray<simd::int4>, primitive<unsigned>> hr;
    auto c = get_color(colors, hr, primitive<unsigned>{}, per_face_binding{});

#elif defined GET_COLOR_MASK4

    vector<3, float> *colors;
    hit_record<basic_ray<simd::mask4>, primitive<unsigned>> hr;
    auto c = get_color(colors, hr, primitive<unsigned>{}, per_face_binding{});

#elif defined GET_COLOR_TRI_FLOAT4

    vector<3, float> *colors;
    hit_record<basic_ray<simd::float4>, primitive<unsigned>> hr;
    vector<3, simd::float4> c = get_color(colors, hr, basic_triangle<3, float, unsigned>{}, per_vertex_binding{});

#elif defined GET_COLOR_TRI_FLOAT8

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
    vector<3, float> *colors;
    hit_record<basic_ray<simd::float8>, primitive<unsigned>> hr;
    vector<3, simd::float8> c = get_color(colors, hr, basic_triangle<3, float, unsigned>{}, per_vertex_binding{});
#endif

#elif defined GET_COLOR_TRI_INT4

    vector<3, float> *colors;
    hit_record<basic_ray<simd::int4>, primitive<unsigned>> hr;
    auto c = get_color(colors, hr, basic_triangle<3, float, unsigned>{}, per_vertex_binding{});

#elif defined GET_COLOR_TRI_MASK4

    vector<3, float> *colors;
    hit_record<basic_ray<simd::mask4>, primitive<unsigned>> hr;
    auto c = get_color(colors, hr, basic_triangle<3, float, unsigned>{}, per_vertex_binding{});

#endif

    return 0;
}
