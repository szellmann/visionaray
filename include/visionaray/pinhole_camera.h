// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_PINHOLE_CAMERA_H
#define VSNRAY_PINHOLE_CAMERA_H 1

#include "math/aabb.h"
#include "math/matrix.h"
#include "math/rectangle.h"
#include "math/vector.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Simple pinhole camera class, similar interface to OpenGL/GLU
//
//
//  - pinhole_camera::look_at()
//      specify a camera by eye and center of view positions and an up direction
//      vector up must be normalized
//
//  - pinhole_camera::perspective()
//      specify the perspective projection transformation of the camera in terms
//      of field of view angle in y direction (radians!), the viewports aspect
//      ratio and the distances from the viewer from the near clipping plane and
//      the far clipping plane
//      NOTE: in contrast to gluPerspective(), the field of view parameter (fovy)
//      must be specified in radians!
//
//  - pinhole_camera::set_viewport()
//      specify the viewport rectangle for the viewing configuration, by
//      parameters x-origin, y-origin (lower left corner), width and height
//
//  - pinhole_camera::view_all()
//      locates the camera at the outside of the given bounding box by calculating
//      the bounding sphere, pushing the camera position to the far side of the
//      sphere along the current viewing direction and looking to the center of
//      the sphere
//
//
//-------------------------------------------------------------------------------------------------

class pinhole_camera
{
public:

    void look_at(vec3 const& eye, vec3 const& center, vec3 const& up = vec3(0.0f, 1.0f, 0.0f));
    void perspective(float fovy, float aspect, float z_near, float z_far);
    void set_viewport(recti const& viewport);
    void set_viewport(int left, int bottom, int width, int height);

    //! Depends on the perspective parameters (fovy) to be set
    void view_all(aabb const& box, vec3 const& up = vec3(0.0f, 1.0f, 0.0f));

    void set_view_matrix(mat4 const& view) { view_ = view; }
    void set_proj_matrix(mat4 const& proj) { proj_ = proj; }

    mat4 const& get_view_matrix() const { return view_; }
    mat4 const& get_proj_matrix() const { return proj_; }

    recti get_viewport() const { return viewport_; }

    float fovy() const { return fovy_; }
    float aspect() const { return aspect_; }
    float z_near() const { return z_near_; }
    float z_far() const { return z_far_; }

    vec3 const& eye() const { return eye_; }
    vec3 const& center() const { return center_; }
    vec3 const& up() const { return up_; }

    float distance() const { return distance_; }

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

    vec3 eye_;
    vec3 center_;
    vec3 up_;

    //! Distance between camera position and center of view
    float distance_;

    float fovy_;
    float aspect_;
    float z_near_;
    float z_far_;

    recti viewport_;

    // Precalculated for rendering
    vec3 U;
    vec3 V;
    vec3 W;
};

} // visionaray

#include "detail/pinhole_camera.inl"

#endif // VSNRAY_PINHOLE_CAMERA_H
