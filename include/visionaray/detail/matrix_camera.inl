// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

inline matrix_camera::matrix_camera(mat4 const& view, mat4 const& proj)
    : view_(view)
    , proj_(proj)
{
}

inline void matrix_camera::set_view_matrix(mat4 const& view)
{
    view_ = view;
}

inline void matrix_camera::set_proj_matrix(mat4 const& proj)
{
    proj_ = proj;
}

inline mat4 const& matrix_camera::get_view_matrix() const
{
    return view_;
}

inline mat4 const& matrix_camera::get_proj_matrix() const
{
    return proj_;
}

} // visionaray
