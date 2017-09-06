// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_PINHOLE_CAMERA_H
#define VSNRAY_PINHOLE_CAMERA_H 1

#include "detail/perspective_camera_base.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Simple pinhole camera class, similar interface to OpenGL/GLU
//-------------------------------------------------------------------------------------------------

class pinhole_camera : public detail::perspective_camera_base
{
public:

    // Generate primary ray at (x,y) (may be a subpixel position).
    template <typename R, typename T = typename R::scalar_type>
    VSNRAY_FUNC
    R primary_ray(R /* */, T const& x, T const& y, T const& width, T const& height) const;
};

} // visionaray

#include "detail/pinhole_camera.inl"

#endif // VSNRAY_PINHOLE_CAMERA_H
