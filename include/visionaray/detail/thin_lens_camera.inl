// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../sampling.h"

namespace visionaray
{

template <typename R, typename Generator, typename T>
VSNRAY_FUNC
inline R thin_lens_camera::primary_ray(
        R          /* */,
        Generator& gen,
        T const&   x,
        T const&   y,
        T const&   width,
        T const&   height
        ) const
{
    R r = pinhole_camera::primary_ray(R{}, x, y, width, height);

    vector<3, T> lens_du(normalize(U));
    vector<3, T> lens_dv(normalize(V));
    vector<3, T> lens_normal = cross(lens_du, lens_dv);

    vector<3, T> p = r.dir * (focal_distance_ / abs(dot(r.dir, lens_normal)));

    auto lens_sample = concentric_sample_disk(gen.next(), gen.next());

    vector<3, T> lens_offset
        = (T(lens_radius_) * lens_sample.x) * lens_du
        + (T(lens_radius_) * lens_sample.y) * lens_dv;

    r.ori += lens_offset;
    r.dir = normalize(p - lens_offset);

    return r;
}

} // visionaray
