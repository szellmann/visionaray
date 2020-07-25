// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_POINT_LIGHT_H
#define VSNRAY_POINT_LIGHT_H 1

#include "detail/macros.h"
#include "math/vector.h"
#include "light_sample.h"

namespace visionaray
{

template <typename T>
class point_light
{
public:

    using scalar_type   = T;
    using vec_type      = vector<3, T>;
    using color_type    = vector<3, T>;

public:

    // Evaluate the light intensity at pos.
    template <typename U>
    VSNRAY_FUNC vector<3, U> intensity(vector<3, U> const& pos) const;

    // Get a single sampled position (always the same).
    template <typename Generator, typename U = typename Generator::value_type>
    VSNRAY_FUNC light_sample<U> sample(Generator& gen) const;

    VSNRAY_FUNC vec_type position() const;
    VSNRAY_FUNC T constant_attenuation() const;
    VSNRAY_FUNC T linear_attenuation() const;
    VSNRAY_FUNC T quadratic_attenuation() const;

    VSNRAY_FUNC void set_cl(color_type const& cl);
    VSNRAY_FUNC void set_kl(scalar_type kl);
    VSNRAY_FUNC void set_position(vec_type const& pos);
    VSNRAY_FUNC void set_constant_attenuation(T att);
    VSNRAY_FUNC void set_linear_attenuation(T att);
    VSNRAY_FUNC void set_quadratic_attenuation(T att);

private:

    color_type  cl_;
    scalar_type kl_;
    vec_type    position_;

    T constant_attenuation_;
    T linear_attenuation_;
    T quadratic_attenuation_;
};

} // visionaray

#include "detail/point_light.inl"

#endif // VSNRAY_POINT_LIGHT_H
