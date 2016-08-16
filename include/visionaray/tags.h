// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TAGS_H
#define VSNRAY_TAGS_H 1

#include <type_traits>

//-------------------------------------------------------------------------------------------------
// Tags for API use
//-------------------------------------------------------------------------------------------------

namespace visionaray
{

struct bvh_tag {};
struct index_bvh_tag {};

struct conductor_tag {};
struct dielectric_tag {};


//-------------------------------------------------------------------------------------------------
// Data binding tags
// Determine how precalculated data from an array is bound to the vertices of a primitive
// In the case of primitives w/o vertices, unspecified_binding applies
//

struct data_binding {};
struct unspecified_binding         : data_binding {};
struct per_face_binding            : data_binding {};
struct per_vertex_binding          : data_binding {};
struct per_geometry_binding        : data_binding {};


//-------------------------------------------------------------------------------------------------
// Explicit data binding types for normals and colors
//

struct color_binding {};
struct colors_per_face_binding     : color_binding {};
struct colors_per_vertex_binding   : color_binding {};
struct colors_per_geometry_binding : color_binding {};

struct normal_binding {};
struct normals_per_face_binding    : normal_binding {};
struct normals_per_vertex_binding  : normal_binding {};


//-------------------------------------------------------------------------------------------------
// Traits to identify bindings
//

template <typename T>
using is_general_binding = std::is_base_of<data_binding, T>;

template <typename T>
using is_color_binding   = std::is_base_of<color_binding, T>;

template <typename T>
using is_normal_binding  = std::is_base_of<normal_binding, T>;

} // visionaray

#endif // VSNRAY_TAGS_H
