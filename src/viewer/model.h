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
namespace detail
{

typedef aligned_vector<basic_triangle<3, float>>                            triangle_list;
typedef aligned_vector<vector<3, float>>                                    normal_list;
typedef aligned_vector<vector<2, float>>                                    tex_coord_list;
typedef aligned_vector<phong<float>>                                        mat_list;
typedef aligned_vector<texture<vector<3, unsigned char>, ElementType, 2>>   tex_list;

struct model
{
    triangle_list   primitives;
    normal_list     normals;
    tex_coord_list  tex_coords;
    mat_list        materials;
    tex_list        textures;
    aabb            bbox;
};

} // detail
} // visionaray

#endif // VSNRAY_VIEWER_MODEL_H
