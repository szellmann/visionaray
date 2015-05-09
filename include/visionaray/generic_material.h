// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GENERIC_MATERIAL_H
#define VSNRAY_GENERIC_MATERIAL_H

#include "detail/macros.h"
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
    /* implicit */ generic_material(M<scalar_type> const& material)
        : base_type(material)
    {
    }

    VSNRAY_FUNC bool is_emissive() const
    {
        return apply_visitor( is_emissive_visitor(), *this );
    }

    VSNRAY_FUNC spectrum<scalar_type> ambient() const
    {
        return apply_visitor( ambient_visitor(), *this );
    }

    template <typename SR>
    VSNRAY_FUNC
    spectrum<typename SR::scalar_type> shade(SR const& sr) const
    {
        return apply_visitor( shade_visitor<SR>(sr), *this );
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<U> sample(SR const& sr, vector<3, U>& refl_dir, U& pdf, S& sampler) const
    {
        return apply_visitor( sample_visitor<SR, U, S>(sr, refl_dir, pdf, sampler), *this );
    }

private:

    // Variant visitors

    struct is_emissive_visitor;

    struct ambient_visitor;

    template <typename SR>
    struct shade_visitor;

    template <typename SR, typename U, typename S>
    struct sample_visitor;

};

} // visionaray

#include "detail/generic_material.inl"

#endif // VSNRAY_GENERIC_MATERIAL_H
