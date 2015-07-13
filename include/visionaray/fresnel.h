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
spectrum<T> fresnel_reflectance(
        conductor_tag       /* */,
        spectrum<T> const&  eta,
        spectrum<T> const&  k,
        T                   cosi
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

} // visionaray

#endif // VSNRAY_FRESNEL_H
