// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DIRECTIONAL_LIGHT_H
#define VSNRAY_DIRECTIONAL_LIGHT_H 1

#include "detail/macros.h"
#include "math/vector.h"
#include "light_sample.h"

namespace visionaray
{

template <typename T>
class directional_light
{
public:

    using scalar_type = T;
    using vec_type    = vector<3, T>;
    using color_type  = vector<3, T>;

public:

    // Evaluata the light intensity 
    template <typename U>
    VSNRAY_FUNC vector<3, U> intensity(vector<3, U> const& pos) const;
    
    // Sample
    template <typename Generator, typename U = typename Generator::value_type>
    VSNRAY_FUNC light_sample<U> sample(vector<3, U> const& reference_point, Generator& gen) const;

    VSNRAY_FUNC vec_type direction() const;
    // 0.0 means delta light source
    VSNRAY_FUNC T angular_diameter() const;

    VSNRAY_FUNC void set_cl(color_type const& cl);
    VSNRAY_FUNC void set_kl(scalar_type kl);
    VSNRAY_FUNC void set_direction(vec_type const& dir);
    // 0.0 means delta light source
    VSNRAY_FUNC void set_angular_diameter(T const & ad);

private:

    color_type  cl_;
    scalar_type kl_;
    vec_type    direction_;
    T           angular_diameter_;
};

} // visionaray

#include "detail/directional_light.inl"

#endif // VSNRAY_DIRECTIONAL_LIGHT_H
