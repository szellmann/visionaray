// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_FORWARD_H
#define VSNRAY_MATH_FORWARD_H 1

#include <cstddef>

#include "config.h"

#undef min
#undef max


namespace MATH_NAMESPACE
{


//--------------------------------------------------------------------------------------------------
// Declarations
//

template <size_t Dim>
class cartesian_axis;

template <unsigned Bits>
class snorm;

template <unsigned Bits>
class unorm;

template <size_t Dim, typename T>
class vector;

template <typename T>
class quaternion;

template <size_t N /* rows */, size_t M /* columns */, typename T>
class matrix;

template <size_t Dim, typename T, typename P = unsigned>
class basic_plane;

template <typename T, size_t Dim = 3>
class basic_aabb;

template <typename T>
class basic_ray;

template <typename T, typename P = unsigned>
class basic_sphere;

template <size_t Dim, typename T, typename P = unsigned>
class basic_triangle;

template <typename Layout, typename T>
class rectangle;

template <typename T>
class xywh_layout;


//--------------------------------------------------------------------------------------------------
// Most common typedefs
//

typedef vector<2, int>                         vec2i;
typedef vector<2, unsigned int>                vec2ui;
typedef vector<2, float>                       vec2f;
typedef vector<2, double>                      vec2d;
typedef vector<2, float>                       vec2;


typedef vector<3, int>                         vec3i;
typedef vector<3, unsigned int>                vec3ui;
typedef vector<3, float>                       vec3f;
typedef vector<3, double>                      vec3d;
typedef vector<3, float>                       vec3;


typedef vector<4, int>                         vec4i;
typedef vector<4, unsigned int>                vec4ui;
typedef vector<4, float>                       vec4f;
typedef vector<4, double>                      vec4d;
typedef vector<4, float>                       vec4;


typedef quaternion<float>                      quatf;
typedef quaternion<double>                     quatd;
typedef quaternion<float>                      quat;


typedef matrix<2, 2, float>                    mat2f;
typedef matrix<2, 2, double>                   mat2d;
typedef matrix<2, 2, float>                    mat2;
typedef matrix<2, 2, float>                    mat2x2f;
typedef matrix<2, 2, double>                   mat2x2d;
typedef matrix<2, 2, float>                    mat2x2;


typedef matrix<3, 3, float>                    mat3f;
typedef matrix<3, 3, double>                   mat3d;
typedef matrix<3, 3, float>                    mat3;
typedef matrix<3, 3, float>                    mat3x3f;
typedef matrix<3, 3, double>                   mat3x3d;
typedef matrix<3, 3, float>                    mat3x3;


typedef matrix<4, 3, float>                    mat4x3f;
typedef matrix<4, 3, double>                   mat4x3d;
typedef matrix<4, 3, float>                    mat4x3;


typedef matrix<4, 4, float>                    mat4f;
typedef matrix<4, 4, double>                   mat4d;
typedef matrix<4, 4, float>                    mat4;
typedef matrix<4, 4, float>                    mat4x4f;
typedef matrix<4, 4, double>                   mat4x4d;
typedef matrix<4, 4, float>                    mat4x4;


typedef basic_plane<3, int>                    plane3i;
typedef basic_plane<3, float>                  plane3f;
typedef basic_plane<3, double>                 plane3d;
typedef basic_plane<3, float>                  plane3;


typedef basic_aabb<int>                        aabbi;
typedef basic_aabb<float>                      aabbf;
typedef basic_aabb<double>                     aabbd;
typedef basic_aabb<float>                      aabb;


typedef basic_ray<float>                       ray;


typedef rectangle<xywh_layout<int>, int>       recti;
typedef rectangle<xywh_layout<float>, float>   rectf;
typedef rectangle<xywh_layout<double>, double> rectd;
typedef rectangle<xywh_layout<float>, float>   rect;


} // MATH_NAMESPACE


#endif // VSNRAY_MATH_FORWARD_H
