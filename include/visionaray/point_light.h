// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_POINT_LIGHT_H
#define VSNRAY_POINT_LIGHT_H

#include "detail/macros.h"
#include "math/math.h"

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

    VSNRAY_FUNC
    color_type color() const
    {
        return cl_ * kl_;
    }

    VSNRAY_FUNC
    vec_type position() const
    {
        return position_;
    }

    VSNRAY_FUNC
    T constant_attenuation() const
    {
        return constant_attenuation_;
    }

    VSNRAY_FUNC
    T linear_attenuation() const
    {
        return linear_attenuation_;
    }

    VSNRAY_FUNC
    T quadratic_attenuation() const
    {
        return quadratic_attenuation_;
    }

    void set_cl(color_type const& cl)
    {
        cl_ = cl;
    }

    void set_kl(scalar_type kl)
    {
        kl_ = kl;
    }

    void set_position(vec_type const& pos)
    {
        position_ = pos;
    }

    void set_constant_attenuation(T att)
    {
        constant_attenuation_ = att;
    }

    void set_linear_attenuation(T att)
    {
        linear_attenuation_ = att;
    }

    void set_quadratic_attenuation(T att)
    {
        quadratic_attenuation_ = att;
    }

private:

    color_type  cl_;
    scalar_type kl_;
    vec_type    position_;

    T constant_attenuation_     = T(1.0);
    T linear_attenuation_       = T(0.0);
    T quadratic_attenuation_    = T(0.0);
};

} // visionaray

#endif // VSNRAY_POINT_LIGHT_H
