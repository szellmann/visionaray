// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_FRESNEL_H
#define VSNRAY_FRESNEL_H 1

#include "detail/macros.h"
#include "spectrum.h"
#include "tags.h"

namespace visionaray
{

template <typename T>
VSNRAY_FUNC
inline spectrum<T> fresnel_reflectance(
        conductor_tag       /* */,
        spectrum<T> const&  eta,
        spectrum<T> const&  k,
        T const&            cosi
        )
{
    // approximation for s-polarized light (perpendicular), squared
    auto rs2 = ( ( eta * eta + k * k ) - T(2.0) * eta * cosi + cosi * cosi )
             / ( ( eta * eta + k * k ) + T(2.0) * eta * cosi + cosi * cosi );

    // approximation for p-polarized light (parallel), squared
    auto rp2 = ( ( eta * eta + k * k ) * cosi * cosi - T(2.0) * eta * cosi + T(1.0) )
             / ( ( eta * eta + k * k ) * cosi * cosi + T(2.0) * eta * cosi + T(1.0) );

    return (rs2 + rp2) / T(2.0);

}

template <typename T>
VSNRAY_FUNC
spectrum<T> fresnel_reflectance(
        dielectric_tag      /* */,
        spectrum<T> const&  etai,
        spectrum<T> const&  etat,
        T                   cosi,
        T                   cost
        )
{
    // approximation for s-polarized light (perpendicular)
    auto rs = ( etai * cosi - etat * cost )
            / ( etai * cosi + etat * cost );

    // approximation for p-polarized light (parallel)
    auto rp = ( etat * cosi - etai * cost )
            / ( etat * cosi + etai * cost );

    return (rs * rs + rp * rp) / T(2.0);
}

} // visionaray

#endif // VSNRAY_FRESNEL_H
