// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SURFACE_H
#define VSNRAY_SURFACE_H

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

template <typename M>
class surface<M>
{
public:

    using scalar_type   = typename M::scalar_type;

public:

    VSNRAY_FUNC surface() = default;

    VSNRAY_FUNC
    surface(vector<3, scalar_type> const& n, M const& m)
        : normal(n)
        , material(m)
    {
    }

    vector<3, scalar_type>  normal;
    M                       material;

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

template <typename M, typename C>
class surface<M, C> : public surface<M>
{
public:

    using base_type     = surface<M>;
    using scalar_type   = typename M::scalar_type;

public:

    VSNRAY_FUNC surface() = default;

    VSNRAY_FUNC
    surface(vector<3, scalar_type> const& n, M const& m, C const& tex_color)
        : base_type(n, m)
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
