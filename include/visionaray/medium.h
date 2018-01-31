// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MEDIUM_H
#define VSNRAY_MEDIUM_H 1

#include "phase_function.h"
#include "spectrum.h"

namespace visionaray
{

template <typename T>
class anisotropic_medium
{
public:

    using scalar_type = T;

public:

    template <typename U>
    VSNRAY_FUNC
    spectrum<U> tr(vector<3, U> const& wo, vector<3, U> const& wi)
    {
        return spectrum<U>(phase_.tr(wo, wi));
    }

    template <typename U, typename Sampler>
    VSNRAY_FUNC
    spectrum<U> sample(vector<3, U> const& wo, vector<3, U>& wi, U& pdf, Sampler& sampler)
    {
        return spectrum<U>(phase_.sample(wo, wi, pdf, sampler));
    }

    // Anisotropy in [-1.0..1.0], where -1.0 scatters all light backwards
    T& anisotropy()
    {
        return phase_.g;
    }

    // Anisotropy in [-1.0..1.0], where -1.0 scatters all light backwards
    VSNRAY_FUNC T const& anisotropy() const
    {
        return phase_.g;
    }

private:

    henyey_greenstein<T> phase_;

};

} // visionaray

#endif // VSNRAY_MEDIUM_H
