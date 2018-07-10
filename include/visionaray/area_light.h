// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_AREA_LIGHT_H
#define VSNRAY_AREA_LIGHT_H 1

#include <cstddef>

#include "detail/macros.h"
#include "math/array.h"
#include "math/vector.h"

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

    VSNRAY_FUNC Geometry& geometry();
    VSNRAY_FUNC Geometry const& geometry() const;

    // Evaluate the light intensity at pos.
    template <typename U>
    VSNRAY_FUNC vector<3, U> intensity(vector<3, U> const& pos) const;

    template <typename U, typename Generator>
    VSNRAY_FUNC vector<3, U> sample(U& pdf, Generator& gen) const;

    // Get N sampled positions.
    template <size_t N, typename U, typename Generator>
    VSNRAY_FUNC void sample(
            array<U, N>& pdfs,
            array<vector<3, U>, N>& result,
            Generator& gen
            ) const;

    // Return center of bounding box
    // TODO: maybe return something more meaningful, e.g. center of gravity?
    VSNRAY_FUNC vector<3, T> position() const;

private:

    Geometry geometry_;

};

} // visionaray

#include "detail/area_light.inl"

#endif // VSNRAY_AREA_LIGHT_H
