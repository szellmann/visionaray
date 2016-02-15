// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CAMERA_H
#define VSNRAY_CAMERA_H 1

#include "math/math.h"

namespace visionaray
{

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
