// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TAGS_H
#define VSNRAY_TAGS_H 1

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
struct unspecified_binding  : data_binding {};
struct per_face_binding     : data_binding {};
struct per_vertex_binding   : data_binding {};
struct per_geometry_binding : data_binding {};

// more explicit versions ---------------------------------

using colors_per_face_binding     = per_face_binding;
using colors_per_vertex_binding   = per_vertex_binding;
using colors_per_geometry_binding = per_geometry_binding;

using normals_per_face_binding    = per_face_binding;
using normals_per_vertex_binding  = per_vertex_binding;

} // visionaray

#endif // VSNRAY_TAGS_H
