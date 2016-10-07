// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SPOT_LIGHT_H
#define VSNRAY_SPOT_LIGHT_H 1

#include "detail/macros.h"
#include "math/vector.h"

namespace visionaray
{

template <typename T>
class spot_light
{
public:

    using scalar_type   = T;
    using vec_type      = vector<3, T>;
    using color_type    = vector<3, T>;

public:

    // Evaluate the light intensity at pos.
    template <typename U>
    VSNRAY_FUNC vector<3, U> intensity(vector<3, U> const& pos) const;

    VSNRAY_FUNC vec_type position() const;
    VSNRAY_FUNC vec_type spot_direction() const;
    VSNRAY_FUNC T spot_cutoff() const;
    VSNRAY_FUNC T spot_exponent() const;
    VSNRAY_FUNC T constant_attenuation() const;
    VSNRAY_FUNC T linear_attenuation() const;
    VSNRAY_FUNC T quadratic_attenuation() const;

    VSNRAY_FUNC void set_cl(color_type const& cl);
    VSNRAY_FUNC void set_kl(scalar_type kl);
    VSNRAY_FUNC void set_position(vec_type const& pos);

    // Set spot direction (must be unit vector!).
    VSNRAY_FUNC void set_spot_direction(vec_type const& dir);

    // Set the spot cutoff (angle in radians!).
    VSNRAY_FUNC void set_spot_cutoff(T cutoff);
    VSNRAY_FUNC void set_spot_exponent(T exponent);
    VSNRAY_FUNC void set_constant_attenuation(T att);
    VSNRAY_FUNC void set_linear_attenuation(T att);
    VSNRAY_FUNC void set_quadratic_attenuation(T att);

private:

    color_type  cl_;
    scalar_type kl_;
    vec_type    position_;
    vec_type    spot_direction_;
    scalar_type spot_cutoff_;
    scalar_type spot_cos_cutoff_;
    scalar_type spot_exponent_;

    T constant_attenuation_     = T(1.0);
    T linear_attenuation_       = T(0.0);
    T quadratic_attenuation_    = T(0.0);
};

} // visionaray

#include "detail/spot_light.inl"

#endif // VSNRAY_SPOT_LIGHT_H
