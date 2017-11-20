// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SURFACE_H
#define VSNRAY_SURFACE_H 1

#include "detail/macros.h"
#include "detail/tags.h"
#include "math/vector.h"
#include "material.h"

namespace visionaray
{

template <typename ...Args>
struct surface;

template <typename N, typename M>
struct surface<N, M>
{
    using scalar_type = typename M::scalar_type;

    N geometric_normal;
    N shading_normal;
    M material;

    template <typename SR>
    VSNRAY_FUNC
    spectrum<scalar_type> shade(SR const& shade_rec)
    {
        return material.shade(shade_rec);
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<scalar_type> sample(SR const& shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler)
    {
        return material.sample(shade_rec, refl_dir, pdf, sampler);
    }
};

template <typename N, typename M, typename C>
struct surface<N, M, C>
{
    using base_type   = surface<N, M>;
    using scalar_type = typename M::scalar_type;

    N geometric_normal;
    N shading_normal;
    M material;
    C tex_color;

    template <typename SR>
    VSNRAY_FUNC
    spectrum<scalar_type> shade(SR shade_rec)
    {
        shade_rec.tex_color = tex_color;
        return material.shade(shade_rec);
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<scalar_type> sample(SR shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler)
    {
        shade_rec.tex_color = tex_color;
        return material.sample(shade_rec, refl_dir, pdf, sampler);
    }
};

} // visionaray

#include "detail/surface.inl"

#endif // VSNRAY_SURFACE_H
