// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_ENVIRONMENT_LIGHT_H
#define VSNRAY_ENVIRONMENT_LIGHT_H 1

#include "detail/macros.h"
#include "math/matrix.h"
#include "spectrum.h"

namespace visionaray
{

template <typename T, typename Texture>
class environment_light
{
public:

    using scalar_type  = T;
    using texture_type = Texture;

public:

    template <typename U>
    VSNRAY_FUNC vector<3, U> intensity(vector<3, U> const& dir) const;

    VSNRAY_FUNC Texture& texture();
    VSNRAY_FUNC Texture const& texture() const;

    VSNRAY_FUNC spectrum<T>& scale();
    VSNRAY_FUNC spectrum<T> const& scale() const;

    // Set light to world transform; also sets world to light (its inverse)
    VSNRAY_FUNC void set_light_to_world_transform(matrix<4, 4, T> const& light_to_world_transform);

    VSNRAY_FUNC matrix<4, 4, T> light_to_world_transform() const;
    VSNRAY_FUNC matrix<4, 4, T> world_to_light_transform() const;

    VSNRAY_FUNC operator bool() const;

private:
    Texture texture_;

    spectrum<T> scale_;

    matrix<4, 4, T> light_to_world_transform_;
    matrix<4, 4, T> world_to_light_transform_;
};

} // visionaray

#include "detail/environment_light.inl"

#endif // VSNRAY_ENVIRONMENT_LIGHT_H
