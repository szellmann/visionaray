// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// area_light members
//

template <typename T, typename Geometry>
inline area_light<T, Geometry>::area_light(Geometry geometry)
    : geometry_(geometry)
{
}

template <typename T, typename Geometry>
template <typename U>
VSNRAY_FUNC
inline vector<3, U> area_light<T, Geometry>::intensity(vector<3, U> const& pos) const
{
    VSNRAY_UNUSED(pos);

    return vector<3, U>(cl_ * kl_);
}

template <typename T, typename Geometry>
template <typename Generator, typename U>
VSNRAY_FUNC
inline light_sample<U> area_light<T, Geometry>::sample(vector<3, U> const& reference_point, Generator& gen) const
{
    light_sample<U> result;

    auto pos = sample_surface(geometry_, reference_point, gen);

    // Satisfy get_normal() interface
    struct { vector<3, U> isect_pos; } hr;
    hr.isect_pos = pos;

    result.dir = pos - reference_point;
    result.dist = length(result.dir);
    result.intensity = intensity(pos);
    result.normal = get_normal(hr, geometry_);
    result.area = U(area(geometry_));
    result.delta_light = false;

    return result;
}

template <typename T, typename Geometry>
VSNRAY_FUNC
inline vector<3, T> area_light<T, Geometry>::position() const
{
    return vector<3, T>(get_bounds(geometry_).center());
}

template <typename T, typename Geometry>
VSNRAY_FUNC
inline Geometry& area_light<T, Geometry>::geometry()
{
    return geometry_;
}

template <typename T, typename Geometry>
VSNRAY_FUNC
inline Geometry const& area_light<T, Geometry>::geometry() const
{
    return geometry_;
}

template <typename T, typename Geometry>
VSNRAY_FUNC
inline void area_light<T, Geometry>::set_cl(vector<3, T> const& cl)
{
    cl_ = cl;
}

template <typename T, typename Geometry>
VSNRAY_FUNC
inline void area_light<T, Geometry>::set_kl(T kl)
{
    kl_ = kl;
}

} // visionaray
