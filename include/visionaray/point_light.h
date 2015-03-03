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

private:

    color_type  cl_;
    scalar_type kl_;
    vec_type    position_;

};

} // visionaray

#endif // VSNRAY_POINT_LIGHT_H
