// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_SPD_BLACKBODY_H
#define VSNRAY_DETAIL_SPD_BLACKBODY_H 1

#include <cmath>

#include "../macros.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Spectral power distribution for blackbody radiator
// See: http://www.spectralcalc.com/blackbody/units_of_wavelength.html
// Color temperature in Kelvin, sample SPD with wavelength (nm)
//

struct blackbody
{
    blackbody(float T = 1500.0) : T(T) {}

    VSNRAY_FUNC float operator()(float lambda /* nm */) const
    {
        double const k = 1.3806488E-23;
        double const h = 6.62606957E-34;
        double const c = 2.99792458E8;

        lambda *= 1E-3; // nm to microns

        return ( ( 2.0 * 1E24 * h * c * c ) / pow(lambda, 5.0) )
             * ( 1.0 / (exp((1E6 * h * c) / (lambda * k * T)) - 1.0) );
    }

private:

    double T;

};

} // visionaray

#endif // VSNRAY_DETAIL_SPD_BLACKBODY_H
