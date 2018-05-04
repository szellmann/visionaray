// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../sampling.h"

namespace visionaray
{

template <typename R, typename Generator, typename T>
VSNRAY_FUNC
inline R thin_lens_camera::primary_ray(
        R         /* */,
        Generator gen,
        T const&  x,
        T const&  y,
        T const&  width,
        T const&  height
        ) const
{
    R r = pinhole_camera::primary_ray(R{}, x, y, width, height);

    auto f = focal_distance_;
    auto p = r.ori + r.dir * T(f);

    auto lens_sample = concentric_sample_disk(gen.next(), gen.next()) * T(lens_radius_);

    r.ori += vector<3, T>(lens_sample.x, lens_sample.y, T(0.0));
    r.dir = normalize(p - r.ori);

    return r;
}

} // visionaray
