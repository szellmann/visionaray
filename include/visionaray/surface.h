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

template <typename M, typename ...Args>
class surface<M, Args...>
{
public:

    using scalar_type   = typename M::scalar_type;

public:

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
    spectrum<scalar_type> shade(SR shade_rec)
    {
        shade_rec.normal = normal;
        return material.shade(shade_rec);
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<scalar_type> sample(SR shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler)
    {
        shade_rec.normal = normal;
        return material.sample(shade_rec, refl_dir, pdf, sampler);
    }

};

template <typename M, typename C, typename ...Args>
class surface<M, C, Args...> : public surface<M, Args...>
{
public:

    using base_type     = surface<M, Args...>;
    using scalar_type   = typename M::scalar_type;

public:

    VSNRAY_FUNC
    surface(vector<3, scalar_type> const& n, M const& m, C const& cd)
        : base_type(n, m)
        , cd_(cd)
    {
    }

    template <typename SR>
    VSNRAY_FUNC
    spectrum<scalar_type> shade(SR shade_rec)
    {
        shade_rec.cd = cd_;
        return base_type::shade(shade_rec);
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<scalar_type> sample(SR shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler)
    {
        shade_rec.cd = cd_;
        return base_type::sample(shade_rec, refl_dir, pdf, sampler);
    }

    C cd_;

};

} // visionaray

#include "detail/surface.inl"

#endif // VSNRAY_SURFACE_H
