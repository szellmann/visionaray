// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MODEL_H
#define VSNRAY_COMMON_MODEL_H 1

#include <map>
#include <string>

#include <visionaray/math/forward.h>
#include <visionaray/texture/forward.h>
#include <visionaray/aligned_vector.h>

namespace visionaray
{

class model
{
public:

    struct material_type
    {
        material_type()
            : name("")
            , ca(0.2f, 0.2f, 0.2f)
            , cd(0.8f, 0.8f, 0.8f)
            , cs(0.1f, 0.1f, 0.1f)
            , ce(0.0f, 0.0f, 0.0f)
            , cr(0.0f, 0.0f, 0.0f)
            , ior(1.0f, 1.0f, 1.0f)
            , absorption(0.0f, 0.0f, 0.0f)
            , specular_exp(32.0f)
            , glossiness(0.0f)
        {
        }

        // Material name.
        std::string name;

        // Ambient color.
        vec3 ca;

        // Diffuse color.
        vec3 cd;

        // Specular color.
        vec3 cs;

        // Emissive color.
        vec3 ce;

        // Reflective color.
        vec3 cr;

        // Index of refraction.
        vec3 ior;

        // Absorption.
        vec3 absorption;

        // Specular exponent.
        float specular_exp;

        // Glossiness term.
        float glossiness;
    };

    using triangle_type     = basic_triangle<3, float>;
    using normal_type       = vector<3, float>;
    using tex_coord_type    = vector<2, float>;
    using texture_type      = texture<vector<4, unorm<8>>, 2>;

    using triangle_list     = aligned_vector<triangle_type>;
    using normal_list       = aligned_vector<normal_type>;
    using tex_coord_list    = aligned_vector<tex_coord_type>;
    using mat_list          = aligned_vector<material_type>;
    using tex_map           = std::map<std::string, texture_type>;
    using tex_list          = aligned_vector<typename texture_type::ref_type>;

public:

    triangle_list   primitives;
    normal_list     shading_normals;
    normal_list     geometric_normals;
    tex_coord_list  tex_coords;
    mat_list        materials;
    tex_map         texture_map;
    tex_list        textures;
    aabb            bbox;

};

} // visionaray

#endif // VSNRAY_COMMON_MODEL_H
