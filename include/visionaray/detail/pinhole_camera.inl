// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <cmath>

namespace visionaray
{

inline void pinhole_camera::look_at(vec3 const& eye, vec3 const& center, vec3 const& up)
{

    eye_      = eye;
    center_   = center;
    up_       = up;
    distance_ = length(eye - center);

    vec3 f = normalize(eye - center);
    vec3 s = normalize(cross(up, f));
    vec3 u = cross(f, s);

    view_ = mat4
    (
        s.x, u.x, f.x, 0.0f,
        s.y, u.y, f.y, 0.0f,
        s.z, u.z, f.z, 0.0f,
        -dot(eye, s), -dot(eye, u), -dot(eye, f), 1.0f
    );

}

inline void pinhole_camera::perspective(float fovy, float aspect, float z_near, float z_far)
{

    assert( z_near > 0.0f );

    fovy_   = fovy;
    aspect_ = aspect;
    z_near_ = z_near;
    z_far_  = z_far;

    float f = cot(fovy * 0.5f);

    proj_(0, 0) = f / aspect;
    proj_(0, 1) = 0.0f;
    proj_(0, 2) = 0.0f;
    proj_(0, 3) = 0.0f;

    proj_(1, 0) = 0.0f;
    proj_(1, 1) = f;
    proj_(1, 2) = 0.0f;
    proj_(1, 3) = 0.0f;

    proj_(2, 0) = 0.0f;
    proj_(2, 1) = 0.0f;
    proj_(2, 2) = (z_far + z_near) / (z_near - z_far);
    proj_(2, 3) = (2.0f * z_far * z_near) / (z_near - z_far);

    proj_(3, 0) = 0.0f;
    proj_(3, 1) = 0.0f;
    proj_(3, 2) = -1.0f;
    proj_(3, 3) = 0.0f;

}

inline void pinhole_camera::set_viewport(recti const& viewport)
{
    viewport_ = viewport;
}

inline void pinhole_camera::set_viewport(int left, int bottom, int width, int height)
{
    viewport_.x = left;
    viewport_.y = bottom;
    viewport_.w = width;
    viewport_.h = height;
}

inline void pinhole_camera::view_all(aabb const& box, vec3 const& up)
{
    float diagonal = length(box.size());
    float r = diagonal * 0.5f;

    vec3 eye = box.center() + vec3(0, 0, r + r / std::atan(fovy_));

    look_at(eye, box.center(), up);
}

  
inline void pinhole_camera::begin_frame()
{
    // front, side, and up vectors form an orthonormal basis
    auto f = normalize(eye_ - center_);
    auto s = normalize(cross(up_, f));
    auto u =           cross(f, s);

    U = s * tan(fovy_ / 2.0f) * aspect_;
    V = u * tan(fovy_ / 2.0f);
    W = -f;
}

inline void pinhole_camera::end_frame()
{
}

template <typename R, typename T>
VSNRAY_FUNC
inline R pinhole_camera::primary_ray(R /* */, T const& x, T const& y, T const& width, T const& height) const
{
    auto u = T(2.0) * (x + T(0.5)) / width  - T(1.0);
    auto v = T(2.0) * (y + T(0.5)) / height - T(1.0);

    R r;
    r.ori = vector<3, T>(eye_);
    r.dir = normalize(vector<3, T>(U) * u + vector<3, T>(V) * v + vector<3, T>(W));
    return r;
}

} // visionaray
