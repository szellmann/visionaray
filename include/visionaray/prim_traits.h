// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_PRIM_TRAITS_H
#define VSNRAY_PRIM_TRAITS_H 1

#include <cstddef>

#include "math/plane.h"
#include "math/sphere.h"
#include "math/triangle.h"

#include "tags.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Primitive traits, can optionally be reimplemented for custom types by the user
// Implemented for builtin types, if applicable
//
//
//  - scalar_type:
//      get the scalar type / floating point type associated with the primitive
//      default: float
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
// scalar_type
//

template <typename Primitive, typename Check = void>
struct scalar_type
{
    using type = float;
};

// specializations ----------------------------------------

template <size_t Dim, typename T, typename P>
struct scalar_type<basic_plane<Dim, T, P>>
{
    using type = T;
};

template <typename T, typename P>
struct scalar_type<basic_sphere<T, P>>
{
    using type = T;
};

template <size_t Dim, typename T, typename P>
struct scalar_type<basic_triangle<Dim, T, P>>
{
    using type = T;
};

//-------------------------------------------------------------------------------------------------
// Number of vertices
//

// general ------------------------------------------------

template <typename Primitive, typename Check = void>
struct num_vertices
{
    enum { value = 0 };
};

// specializations ----------------------------------------

template <size_t Dim, typename T, typename P>
struct num_vertices<basic_triangle<Dim, T, P>>
{
    enum { value = 3 };
};


//-------------------------------------------------------------------------------------------------
// Number of precalculated normals
//

// general ------------------------------------------------

template <typename Primitive, typename NormalBinding, typename Check = void>
struct num_normals
{
    enum { value = 0 };
};

// specializations ----------------------------------------

template <size_t Dim, typename T, typename P>
struct num_normals<basic_triangle<Dim, T, P>, normals_per_face_binding>
{
    enum { value = 1 };
};

template <size_t Dim, typename T, typename P>
struct num_normals<basic_triangle<Dim, T, P>, normals_per_vertex_binding>
{
    enum { value = 3 };
};


//-------------------------------------------------------------------------------------------------
// Number of texture coordinates
//

// general ------------------------------------------------

template <typename Primitive, typename Check = void>
struct num_tex_coords
{
    enum { value = 0 };
};

// specializations ----------------------------------------

template <size_t Dim, typename T, typename P>
struct num_tex_coords<basic_triangle<Dim, T, P>>
{
    enum { value = 3 };
};

} // visionaray

#endif // VSNRAY_PRIM_TRAITS_H
