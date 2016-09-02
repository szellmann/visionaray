// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CAMERA_H
#define VSNRAY_CAMERA_H 1

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
//  - camera::look_at()
//      specify a camera by eye and center of view positions and an up direction
//      vector up must be normalized
//
//  - camera::perspective()
//      specify the perspective projection transformation of the camera in terms
//      of field of view angle in y direction (radians!), the viewports aspect
//      ratio and the distances from the viewer from the near clipping plane and
//      the far clipping plane
//      NOTE: in contrast to gluPerspective(), the field of view parameter (fovy)
//      must be specified in radians!
//
//  - camera::set_viewport()
//      specify the viewport rectangle for the viewing configuration, by
//      parameters x-origin, y-origin (lower left corner), width and height
//
//  - camera::view_all()
//      locates the camera at the outside of the given bounding box by calculating
//      the bounding sphere, pushing the camera position to the far side of the
//      sphere along the current viewing direction and looking to the center of
//      the sphere
//
//
//-------------------------------------------------------------------------------------------------

class camera
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
};

} // visionaray

#include "detail/camera.inl"

#endif // VSNRAY_CAMERA_H
