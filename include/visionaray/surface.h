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
class surface;

template <typename ...Args>
class surface;

template <typename N, typename M>
class surface<N, M>
{
public:

    using scalar_type   = typename M::scalar_type;

public:

    VSNRAY_FUNC surface() = default;

    VSNRAY_FUNC
    surface(N const& gn, N const& sn, M const& m)
        : normal(gn)
        , shading_normal(sn)
        , material(m)
    {
    }

    N normal;
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
class surface<N, M, C> : public surface<N, M>
{
public:

    using base_type     = surface<N, M>;
    using scalar_type   = typename M::scalar_type;

public:

    VSNRAY_FUNC surface() = default;

    VSNRAY_FUNC
    surface(N const& gn, N const& sn, M const& m, C const& tex_color)
        : base_type(gn, sn, m)
        , tex_color_(tex_color)
    {
    }

    template <typename SR>
    VSNRAY_FUNC
    spectrum<scalar_type> shade(SR shade_rec)
    {
        shade_rec.tex_color = tex_color_;
        return base_type::shade(shade_rec);
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<scalar_type> sample(SR shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler)
    {
        shade_rec.tex_color = tex_color_;
        return base_type::sample(shade_rec, refl_dir, pdf, sampler);
    }

    C tex_color_;

};

} // visionaray

#include "detail/surface.inl"

#endif // VSNRAY_SURFACE_H
