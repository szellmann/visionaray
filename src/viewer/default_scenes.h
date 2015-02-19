// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VIEWER_DEFAULT_SCENE_H
#define VSNRAY_VIEWER_DEFAULT_SCENE_H

#include <chrono>
#include <tuple>
#include <vector>

#include <visionaray/detail/aligned_vector.h>
#include <visionaray/math/math.h>
#include <visionaray/generic_prim.h>
#include <visionaray/material.h>

namespace visionaray
{
namespace detail
{

typedef aligned_vector<generic_prim>                generic_prim_list;
typedef aligned_vector<basic_triangle<3, float>>    triangle_list;
typedef aligned_vector<vector<3, float>>            normal_list;
typedef aligned_vector<phong<float>>                phong_mat_list;

typedef std::tuple<generic_prim_list, normal_list, phong_mat_list, aabb> generic_prim_scene;
typedef std::tuple<triangle_list, normal_list, phong_mat_list, aabb> triangle_scene;


generic_prim_scene default_generic_prim_scene()
{
    static const size_t N = 14;

    typedef basic_sphere<float>         sphere_type;
    typedef triangle_list::value_type   triangle_type;

    generic_prim_list primitives(N);
    normal_list normals(N);
    phong_mat_list materials(4);

    triangle_type triangles[N];
    triangles[ 0].v1 = vec3(-1, -1,  1);
    triangles[ 0].e1 = vec3( 1, -1,  1) - triangles[ 0].v1;
    triangles[ 0].e2 = vec3( 0,  1,  0) - triangles[ 0].v1;

    triangles[ 1].v1 = vec3( 1, -1,  1);
    triangles[ 1].e1 = vec3( 1, -1, -1) - triangles[ 1].v1;
    triangles[ 1].e2 = vec3( 0,  1,  0) - triangles[ 1].v1;

    triangles[ 2].v1 = vec3( 1, -1, -1);
    triangles[ 2].e1 = vec3(-1, -1, -1) - triangles[ 2].v1;
    triangles[ 2].e2 = vec3( 0,  1,  0) - triangles[ 2].v1;

    triangles[ 3].v1 = vec3(-1, -1, -1);
    triangles[ 3].e1 = vec3(-1, -1,  1) - triangles[ 3].v1;
    triangles[ 3].e2 = vec3( 0,  1,  0) - triangles[ 3].v1;

    triangles[ 4].v1 = vec3( 1, -1,  1);
    triangles[ 4].e1 = vec3(-1, -1,  1) - triangles[ 4].v1;
    triangles[ 4].e2 = vec3( 1, -1, -1) - triangles[ 4].v1;

    triangles[ 5].v1 = vec3(-1, -1,  1);
    triangles[ 5].e1 = vec3(-1, -1, -1) - triangles[ 5].v1;
    triangles[ 5].e2 = vec3( 1, -1, -1) - triangles[ 5].v1;

    // 2nd pyramid
    triangles[ 6].v1 = vec3(0.3, 0.3, 0.7);
    triangles[ 6].e1 = vec3(0.7, 0.3, 0.7) - triangles[ 6].v1;
    triangles[ 6].e2 = vec3(0.5, 0.7, 0.5) - triangles[ 6].v1;

    triangles[ 7].v1 = vec3(0.7, 0.3, 0.7);
    triangles[ 7].e1 = vec3(0.7, 0.3, 0.3) - triangles[ 7].v1;
    triangles[ 7].e2 = vec3(0.5, 0.7, 0.5) - triangles[ 7].v1;

    triangles[ 8].v1 = vec3(0.7, 0.3, 0.3);
    triangles[ 8].e1 = vec3(0.3, 0.3, 0.3) - triangles[ 8].v1;
    triangles[ 8].e2 = vec3(0.5, 0.7, 0.5) - triangles[ 8].v1;

    triangles[ 9].v1 = vec3(0.3, 0.3, 0.3);
    triangles[ 9].e1 = vec3(0.3, 0.3, 0.7) - triangles[ 9].v1;
    triangles[ 9].e2 = vec3(0.5, 0.7, 0.5) - triangles[ 9].v1;

    triangles[10].v1 = vec3(0.7, 0.3, 0.7);
    triangles[10].e1 = vec3(0.3, 0.3, 0.7) - triangles[10].v1;
    triangles[10].e2 = vec3(0.7, 0.3, 0.3) - triangles[10].v1;

    triangles[11].v1 = vec3(0.3, 0.3, 0.7);
    triangles[11].e1 = vec3(0.3, 0.3, 0.3) - triangles[11].v1;
    triangles[11].e2 = vec3(0.7, 0.3, 0.3) - triangles[11].v1;


    for (size_t i = 0; i < N - 2; ++i)
    {
        triangles[i].prim_id = static_cast<unsigned>(i);
        triangles[i].geom_id = i < 6 ? 0 : 1;
        normals[i] = normalize( cross(triangles[i].e1, triangles[i].e2) );
        primitives[i] = triangles[i];
    }

    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    auto secs = duration_cast<milliseconds>(now.time_since_epoch()).count();

    static float y = -0.5f;
    static float m = 0.1f;
    static const int interval = 1;

    if (secs % interval == 0)
    {
        y += 0.2f * m;

        if (y < -0.5f)
        {
            m = 0.1f;
        }
        else if (y > 1.0f)
        {
            m = -0.1f;
        }
    }

    sphere_type s1;
    s1.prim_id = N - 2;
    s1.geom_id = 2;
    s1.center = vec3(-0.7f, y, 0.8);
    s1.radius = 0.5f;
    primitives[N - 2] = s1;

    sphere_type s2;
    s2.prim_id = N - 1;
    s2.geom_id = 3;
    s2.center = vec3(1.0f, 0.8, -.0f);
    s2.radius = 0.3f;
    primitives[N - 1] = s2;


    for (size_t i = 0; i < 4; ++i)
    {
        phong<float> m;
        if (i == 0)
        {
            m.set_cd( vec3(0.0f, 1.0f, 1.0f) );
        }
        else if (i == 1)
        {
            m.set_cd( vec3(1.0f, 0.0f, 0.0f) );
        }
        else if (i == 2)
        {
            m.set_cd( vec3(0.0f, 0.0f, 1.0f) );
        }
        else
        {
            m.set_cd( vec3(1.0f, 1.0f, 1.0f) );
        }
        m.set_kd( 1.0f );
        m.set_ks( 1.0f );
        m.set_specular_exp( 32.0f );
        materials[i] = m;
    }

    return make_tuple(primitives, normals, materials, aabb(vec3(-1, -1, 0), vec3(1, 1, 2)));
}

triangle_scene default_triangle_scene()
{
    static const size_t N = 12;

    triangle_list triangles(N);
    normal_list normals(N);
    phong_mat_list materials(N);

    triangles[ 0].v1 = vec3(-1, -1,  1);
    triangles[ 0].e1 = vec3( 1, -1,  1) - triangles[ 0].v1;
    triangles[ 0].e2 = vec3( 0,  1,  0) - triangles[ 0].v1;

    triangles[ 1].v1 = vec3( 1, -1,  1);
    triangles[ 1].e1 = vec3( 1, -1, -1) - triangles[ 1].v1;
    triangles[ 1].e2 = vec3( 0,  1,  0) - triangles[ 1].v1;

    triangles[ 2].v1 = vec3( 1, -1, -1);
    triangles[ 2].e1 = vec3(-1, -1, -1) - triangles[ 2].v1;
    triangles[ 2].e2 = vec3( 0,  1,  0) - triangles[ 2].v1;

    triangles[ 3].v1 = vec3(-1, -1, -1);
    triangles[ 3].e1 = vec3(-1, -1,  1) - triangles[ 3].v1;
    triangles[ 3].e2 = vec3( 0,  1,  0) - triangles[ 3].v1;

    triangles[ 4].v1 = vec3( 1, -1,  1);
    triangles[ 4].e1 = vec3(-1, -1,  1) - triangles[ 4].v1;
    triangles[ 4].e2 = vec3( 1, -1, -1) - triangles[ 4].v1;

    triangles[ 5].v1 = vec3(-1, -1,  1);
    triangles[ 5].e1 = vec3(-1, -1, -1) - triangles[ 5].v1;
    triangles[ 5].e2 = vec3( 1, -1, -1) - triangles[ 5].v1;

    // 2nd pyramid
    triangles[ 6].v1 = vec3(0.3, 0.3, 0.7);
    triangles[ 6].e1 = vec3(0.7, 0.3, 0.7) - triangles[ 6].v1;
    triangles[ 6].e2 = vec3(0.5, 0.7, 0.5) - triangles[ 6].v1;

    triangles[ 7].v1 = vec3(0.7, 0.3, 0.7);
    triangles[ 7].e1 = vec3(0.7, 0.3, 0.3) - triangles[ 7].v1;
    triangles[ 7].e2 = vec3(0.5, 0.7, 0.5) - triangles[ 7].v1;

    triangles[ 8].v1 = vec3(0.7, 0.3, 0.3);
    triangles[ 8].e1 = vec3(0.3, 0.3, 0.3) - triangles[ 8].v1;
    triangles[ 8].e2 = vec3(0.5, 0.7, 0.5) - triangles[ 8].v1;

    triangles[ 9].v1 = vec3(0.3, 0.3, 0.3);
    triangles[ 9].e1 = vec3(0.3, 0.3, 0.7) - triangles[ 9].v1;
    triangles[ 9].e2 = vec3(0.5, 0.7, 0.5) - triangles[ 9].v1;

    triangles[10].v1 = vec3(0.7, 0.3, 0.7);
    triangles[10].e1 = vec3(0.3, 0.3, 0.7) - triangles[10].v1;
    triangles[10].e2 = vec3(0.7, 0.3, 0.3) - triangles[10].v1;

    triangles[11].v1 = vec3(0.3, 0.3, 0.7);
    triangles[11].e1 = vec3(0.3, 0.3, 0.3) - triangles[11].v1;
    triangles[11].e2 = vec3(0.7, 0.3, 0.3) - triangles[11].v1;


    for (size_t i = 0; i < N; ++i)
    {
        triangles[i].prim_id = static_cast<unsigned>(i);
        triangles[i].geom_id = i < 6 ? 0 : 1;
        normals[i] = normalize( cross(triangles[i].e1, triangles[i].e2) );
    }

    materials[0].set_cd( vec3(0.0f, 1.0f, 1.0f) );
    materials[0].set_kd( 1.0f );
    materials[0].set_ks( 1.0f );
    materials[0].set_specular_exp( 32.0f );

    materials[1].set_cd( vec3(1.0f, 0.0f, 0.0f) );
    materials[1].set_kd( 1.0f );
    materials[1].set_ks( 1.0f );
    materials[1].set_specular_exp( 32.0f );

    return make_tuple(triangles, normals, materials, aabb(vec3(-1, -1, 0), vec3(1, 1, 2)));
}

} // detail
} // visionaray

#endif // VSNRAY_VIEWER_DEFAULT_SCENE_H


