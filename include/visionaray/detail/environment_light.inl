// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

template <typename T, typename Texture>
template <typename U>
VSNRAY_FUNC vector<3, U> environment_light<T, Texture>::intensity(vector<3, U> const& dir) const
{
    vector<3, U> d = (matrix<4, 4, U>(world_to_light_transform_) * vector<4, U>(dir, U(0.0))).xyz();

    auto x = atan2(d.x, d.z);
    x = select(x < U(0.0), x + constants::two_pi<U>(), x);
    auto y = acos(d.y);

    auto u = x / constants::two_pi<U>();
    auto v = y * constants::inv_pi<U>();

    vector<2, U> tc(u, v);

    return tex2D(texture_, tc).xyz() * vector<3, U>(to_rgb(scale_));
}

template <typename T, typename Texture>
template <typename U>
VSNRAY_FUNC vector<3, U> environment_light<T, Texture>::background_intensity(vector<3, U> const& dir) const
{
    return intensity(dir);
}

template <typename T, typename Texture>
VSNRAY_FUNC
Texture& environment_light<T, Texture>::texture()
{
    return texture_;
}

template <typename T, typename Texture>
VSNRAY_FUNC
Texture const& environment_light<T, Texture>::texture() const
{
    return texture_;
}

template <typename T, typename Texture>
VSNRAY_FUNC
spectrum<T>& environment_light<T, Texture>::scale()
{
    return scale_;
}

template <typename T, typename Texture>
VSNRAY_FUNC
spectrum<T> const& environment_light<T, Texture>::scale() const
{
    return scale_;
}

template <typename T, typename Texture>
VSNRAY_FUNC
void environment_light<T, Texture>::set_light_to_world_transform(matrix<4, 4, T> const& light_to_world_transform)
{
    light_to_world_transform_ = light_to_world_transform;
    world_to_light_transform_ = inverse(light_to_world_transform);
}

template <typename T, typename Texture>
VSNRAY_FUNC
matrix<4, 4, T> environment_light<T, Texture>::light_to_world_transform() const
{
    return light_to_world_transform_;
}

template <typename T, typename Texture>
VSNRAY_FUNC
matrix<4, 4, T> environment_light<T, Texture>::world_to_light_transform() const
{
    return world_to_light_transform_;
}

template <typename T, typename Texture>
VSNRAY_FUNC
environment_light<T, Texture>::operator bool() const
{
    return static_cast<bool>(texture_);
}

} // visionaray
