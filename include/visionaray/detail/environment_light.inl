// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

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
