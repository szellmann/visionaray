// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VIEWER_MODEL_H
#define VSNRAY_VIEWER_MODEL_H

#include <visionaray/detail/aligned_vector.h>
#include <visionaray/math/forward.h>
#include <visionaray/texture/forward.h>
#include <visionaray/material.h>

namespace visionaray
{

class model
{
public:

    using triangle_list     = aligned_vector<basic_triangle<3, float>>;
    using normal_list       = aligned_vector<vector<3, float>>;
    using tex_coord_list    = aligned_vector<vector<2, float>>;
    using mat_list          = aligned_vector<phong<float>>;
    using tex_list          = aligned_vector<texture<vector<3, unsigned char>, ElementType, 2>>;

public:

    triangle_list   primitives;
    normal_list     normals;
    tex_coord_list  tex_coords;
    mat_list        materials;
    tex_list        textures;
    aabb            bbox;

};

} // visionaray

#endif // VSNRAY_VIEWER_MODEL_H
