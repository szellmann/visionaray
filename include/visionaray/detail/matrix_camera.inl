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

inline void matrix_camera::begin_frame()
{
    view_inv_ = inverse(view_);
    proj_inv_ = inverse(proj_);
}

inline void matrix_camera::end_frame()
{
}

template <typename R, typename T>
VSNRAY_FUNC
inline R matrix_camera::primary_ray(R /* */, T const& x, T const& y, T const& width, T const& height) const
{
    auto u = T(2.0) * (x + T(0.5)) / width  - T(1.0);
    auto v = T(2.0) * (y + T(0.5)) / height - T(1.0);

    auto o = matrix<4, 4, T>(view_inv_) * ( matrix<4, 4, T>(proj_inv_) * vector<4, T>(u, v, -1,  1) );
    auto d = matrix<4, 4, T>(view_inv_) * ( matrix<4, 4, T>(proj_inv_) * vector<4, T>(u, v,  1,  1) );

    R r;
    r.ori =            o.xyz() / o.w;
    r.dir = normalize( d.xyz() / d.w - r.ori );
    return r;
}

} // visionaray
