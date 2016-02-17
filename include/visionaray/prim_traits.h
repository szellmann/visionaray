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
// Primitive traits, can optionally be reimplemented for custom types
// Implemented for builtin types, if applicable
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Number of vertices
//

template <typename Primitive>
struct num_vertices
{
    enum { value = 0 };
};

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

template <typename Primitive, typename NormalBinding>
struct num_normals
{
    enum { value = 0 };
};

template <size_t Dim, typename T>
struct num_normals<basic_triangle<Dim, T>, normals_per_face_binding>
{
    enum { value = 1 };
};

template <size_t Dim, typename T>
struct num_normals<basic_triangle<Dim, T>, normals_per_vertex_binding>
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

template <typename Primitive>
struct num_tex_coords
{
    enum { value = 0 };
};

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
