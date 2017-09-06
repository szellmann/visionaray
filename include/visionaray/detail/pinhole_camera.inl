// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

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
