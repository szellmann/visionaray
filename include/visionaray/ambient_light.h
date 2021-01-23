// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_AMBIENT_LIGHT_H
#define VSNRAY_AMBIENT_LIGHT_H 1

#include "math/vector.h"

namespace visionaray
{

template <typename T>
class ambient_light
{
public:

    using scalar_type = T;

public:

    template <typename U>
    VSNRAY_FUNC vector<3, U> intensity(vector<3, U> const& dir) const;

    VSNRAY_FUNC void set_cl(vector<3, T> const& cl);
    VSNRAY_FUNC void set_kl(T const& kl);

private:
    vector<3, T> cl_;
    T kl_;

};

} // visionaray

#include "detail/ambient_light.inl"

#endif // VSNRAY_AMBIENT_LIGHT
