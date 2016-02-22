// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_PRIM_TRAITS_H
#define VSNRAY_PRIM_TRAITS_H 1

#include <visionaray/math/sphere.h>
#include <visionaray/math/triangle.h>

#include <visionaray/tags.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Primitive traits, can optionally be reimplemented for custom types by the user
// Implemented for builtin types, if applicable
//
//
//  - num_vertices:
//      get the vertex count of a primitive, if this primitive is made up of vertices
//      or has control vertices
//      default: value := 0
//
//  - num_normals
//      get the number of precalculated normals that a primitive requires
//      zero if the type requires that normals are calculated on the fly
//      default: value := 0
//
//  - num_tex_coords
//      get the number of precalculated texture coordinates that a primitive requires
//      zero if the type requires that texture coordinates are calculated on the fly
//      default: value := 0
//
//
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Number of vertices
//

// general ------------------------------------------------

template <typename Primitive>
struct num_vertices
{
    enum { value = 0 };
};

// specializations ----------------------------------------

template <size_t Dim, typename T>
struct num_vertices<basic_triangle<Dim, T>>
{
    enum { value = 3 };
};

template <template <typename> class Accelerator, typename Primitive>
struct num_vertices<Accelerator<Primitive>> : num_vertices<Primitive>
{
};


//-------------------------------------------------------------------------------------------------
// Number of precalculated normals
//

// general ------------------------------------------------

template <typename Primitive, typename NormalBinding>
struct num_normals
{
    enum { value = 0 };
};

// specializations ----------------------------------------

template <size_t Dim, typename T>
struct num_normals<basic_triangle<Dim, T>, per_face_binding>
{
    enum { value = 1 };
};

template <size_t Dim, typename T>
struct num_normals<basic_triangle<Dim, T>, per_vertex_binding>
{
    enum { value = 3 };
};

template <template <typename> class Accelerator, typename Primitive, typename NormalBinding>
struct num_normals<Accelerator<Primitive>, NormalBinding> : num_normals<Primitive, NormalBinding>
{
};


//-------------------------------------------------------------------------------------------------
// Number of texture coordinates
//

// general ------------------------------------------------

template <typename Primitive>
struct num_tex_coords
{
    enum { value = 0 };
};

// specializations ----------------------------------------

template <size_t Dim, typename T>
struct num_tex_coords<basic_triangle<Dim, T>>
{
    enum { value = 3 };
};

template <template <typename> class Accelerator, typename Primitive>
struct num_tex_coords<Accelerator<Primitive>> : num_tex_coords<Primitive>
{
};

} // visionaray

#endif // VSNRAY_PRIM_TRAITS_H
