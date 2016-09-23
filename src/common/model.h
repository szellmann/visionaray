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
#include <visionaray/generic_material.h>

namespace visionaray
{

class model
{
public:

    using triangle_type     = basic_triangle<3, float>;
    using normal_type       = vector<3, float>;
    using tex_coord_type    = vector<2, float>;
    using material_type     = plastic<float>;
    using texture_type      = texture<vector<3, unorm<8>>, 2>;

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
