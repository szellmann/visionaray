// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TAGS_H
#define VSNRAH_TAGS_H 1


//-------------------------------------------------------------------------------------------------
// Tags for API use
//

namespace visionaray
{

struct bvh_tag {};
struct index_bvh_tag {};

struct conductor_tag {};
struct dielectric_tag {};

struct colors_binding {};
struct colors_per_face_binding : colors_binding {};
struct colors_per_vertex_binding : colors_binding {};

struct normals_binding {};
struct normals_per_face_binding : normals_binding {};
struct normals_per_vertex_binding : normals_binding {};

} // visionaray

#endif // VSNRAY_TAGS_H
