// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATRIX_CAMERA
#define VSNRAY_MATRIX_CAMERA 1

#include "detail/macros.h"
#include "math/forward.h"
#include "math/matrix.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Camera class that internally stores view and projection matrices
//
// When this camera generates primary rays, it will do so by starting in OpenGL-like NDC,
// with rays that are perpendicular to the image plane, and transforms them to world
// coordinates using the inverse of the internal matrices
//

class matrix_camera
{
public:

    matrix_camera() = default;
    matrix_camera(mat4 const& view, mat4 const& proj);

    void set_view_matrix(mat4 const& view);
    void set_proj_matrix(mat4 const& proj);

    VSNRAY_FUNC
    mat4 const& get_view_matrix() const;

    VSNRAY_FUNC
    mat4 const& get_proj_matrix() const;

    // Call before rendering.
    void begin_frame();

    // Call after rendering.
    void end_frame();

    // Generate primary ray at (x,y) (may be a subpixel position).
    template <typename R, typename T = typename R::scalar_type>
    VSNRAY_FUNC
    R primary_ray(R /* */, T const& x, T const& y, T const& width, T const& height) const;

private:

    mat4 view_;
    mat4 proj_;
    mat4 view_inv_;
    mat4 proj_inv_;

};

} // visionaray

#include "detail/matrix_camera.inl"

#endif
