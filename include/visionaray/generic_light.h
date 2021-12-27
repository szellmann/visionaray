// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GENERIC_LIGHT_H
#define VSNRAY_GENERIC_LIGHT_H 1

#include "detail/macros.h"
#include "math/vector.h"
#include "light_sample.h"
#include "variant.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Generic light
//

template <typename ...Ts>
class generic_light;

template <typename T, typename ...Ts>
class generic_light<T, Ts...> : public variant<T, Ts...>
{
public:

    using scalar_type   = typename T::scalar_type;
    using vec_type      = vector<3, typename T::scalar_type>;
    using color_type    = vector<3, typename T::scalar_type>;
    using base_type     = variant<T, Ts...>;

public:

    generic_light() = default;

    template <typename L>
    /* implicit */ generic_light(L const& light);

    // Evaluate the light intensity at pos.
    template <typename U>
    VSNRAY_FUNC vector<3, U> intensity(vector<3, U> const& pos) const;

    // Get a single sampled position (always the same).
    template <typename Generator, typename U = typename Generator::value_type>
    VSNRAY_FUNC light_sample<U> sample(vector<3, U> const& reference_point, Generator& gen) const;

    // Get the light position.
    VSNRAY_FUNC vector<3, typename T::scalar_type> position() const;

private:

    template <typename U>
    struct intensity_visitor;

    template <typename Generator, typename U = typename Generator::value_type>
    struct sample_visitor;

    struct position_visitor;

};

} // visionaray

#include "detail/generic_light.inl"

#endif // VSNRAY_GENERIC_LIGHT_H
