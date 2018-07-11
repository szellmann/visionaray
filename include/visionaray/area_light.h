// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_AREA_LIGHT_H
#define VSNRAY_AREA_LIGHT_H 1

#include "detail/macros.h"
#include "math/vector.h"
#include "light_sample.h"

namespace visionaray
{

template <typename T, typename Geometry>
class area_light
{
public:

    using scalar_type   = T;
    using vec_type      = vector<3, T>;
    using color_type    = vector<3, T>;

public:

    area_light() = default;
    area_light(Geometry geometry);

    // Evaluate the light intensity at pos.
    template <typename U>
    VSNRAY_FUNC vector<3, U> intensity(vector<3, U> const& pos) const;

    template <typename Generator, typename U = typename Generator::value_type>
    VSNRAY_FUNC light_sample<U> sample(Generator& gen) const;

    // Return center of bounding box
    // TODO: maybe return something more meaningful, e.g. center of gravity?
    VSNRAY_FUNC vector<3, T> position() const;

    VSNRAY_FUNC Geometry& geometry();
    VSNRAY_FUNC Geometry const& geometry() const;

    VSNRAY_FUNC void set_cl(color_type const& cl);
    VSNRAY_FUNC void set_kl(scalar_type kl);

private:

    color_type  cl_;
    scalar_type kl_;

    Geometry geometry_;

};

} // visionaray

#include "detail/area_light.inl"

#endif // VSNRAY_AREA_LIGHT_H
