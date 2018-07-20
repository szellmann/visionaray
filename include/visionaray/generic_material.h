// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GENERIC_MATERIAL_H
#define VSNRAY_GENERIC_MATERIAL_H 1

#include "detail/macros.h"
#include "math/vector.h"
#include "spectrum.h"
#include "variant.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Generic material
//

template <typename ...Ts>
class generic_material;

template <typename T, typename ...Ts>
class generic_material<T, Ts...> : public variant<T, Ts...>
{
public:

    using scalar_type   = typename T::scalar_type;
    using base_type     = variant<T, Ts...>;

public:

    generic_material() = default;

    template <template <typename> class M>
    /* implicit */ generic_material(M<scalar_type> const& material);

    VSNRAY_FUNC spectrum<scalar_type> ambient() const;

    template <typename SR>
    VSNRAY_FUNC spectrum<typename SR::scalar_type> shade(SR const& sr) const;

    template <typename SR, typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC spectrum<U> sample(
            SR const&       sr,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Interaction&    inter,
            Generator&      gen
            ) const;

    template <typename SR, typename Interaction>
    VSNRAY_FUNC typename SR::scalar_type pdf(
            SR const& sr,
            Interaction const& inter
            ) const;

private:

    // Variant visitors

    struct ambient_visitor;

    template <typename SR>
    struct shade_visitor;

    template <typename SR, typename U, typename Interaction, typename Generator>
    struct sample_visitor;

    template <typename SR, typename Interaction>
    struct pdf_visitor;

};

} // visionaray

#include "detail/generic_material.inl"

#endif // VSNRAY_GENERIC_MATERIAL_H
