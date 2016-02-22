// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TAGS_H
#define VSNRAH_TAGS_H 1

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
// Determine how precalculated from an array is bound to the vertices of a primitive
// In the case of primitives w/o vertices, unspecified_binding applies
//

struct data_binding {};
struct unspecified_binding  : data_binding {};
struct per_face_binding     : data_binding {};
struct per_vertex_binding   : data_binding {};

} // visionaray

#endif // VSNRAY_TAGS_H
