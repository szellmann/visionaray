// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATRIX_CAMERA
#define VSNRAY_MATRIX_CAMERA 1

#include "math/matrix.h"

namespace visionaray
{

class matrix_camera
{
public:

    matrix_camera() = default;
    matrix_camera(mat4 const& view, mat4 const& proj);

    void set_view_matrix(mat4 const& view);
    void set_proj_matrix(mat4 const& proj);

    mat4 const& get_view_matrix() const;
    mat4 const& get_proj_matrix() const;

private:

    mat4 view_;
    mat4 proj_;

};

} // visionaray

#include "detail/matrix_camera.inl"

#endif
