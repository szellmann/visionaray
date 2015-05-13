// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_COLOR_CONVERSION_H
#define VSNRAY_DETAIL_COLOR_CONVERSION_H

#include <visionaray/math/math.h>
#include <visionaray/spectrum.h>

#include "spd/d65.h"
#include "macros.h"


namespace visionaray
{

//--------------------------------------------------------------------------------------------------
// CIE 1931 color matching functions
//
// From:
// https://research.nvidia.com/publication/simple-analytic-approximations-cie-xyz-color-matching-functions
//

VSNRAY_FUNC
inline float cie_x(float lambda)
{
    float t1 = (lambda - 442.0f) * ((lambda < 442.0f) ? 0.0624f : 0.0374f);
    float t2 = (lambda - 599.8f) * ((lambda < 599.8f) ? 0.0264f : 0.0323f);
    float t3 = (lambda - 501.1f) * ((lambda < 501.1f) ? 0.0490f : 0.0382f);

    return 0.362f * expf(-0.5f * t1 * t1) + 1.056f * expf(-0.5f * t2 * t2) - 0.065f * expf(-0.5f * t3 * t3);
}

VSNRAY_FUNC
inline float cie_y(float lambda)
{
    float t1 = (lambda - 568.8f) * ((lambda < 568.8f) ? 0.0213f : 0.0247f);
    float t2 = (lambda - 530.9f) * ((lambda < 530.9f) ? 0.0613f : 0.0322f);

    return 0.821f * expf(-0.5f * t1 * t1) + 0.286f * expf(-0.5f * t2 * t2);
}

VSNRAY_FUNC
inline float cie_z(float lambda)
{
    float t1 = (lambda - 437.0f) * ((lambda < 437.0f) ? 0.0845f : 0.0278f);
    float t2 = (lambda - 459.0f) * ((lambda < 459.0f) ? 0.0385f : 0.0725f);

    return 1.217f * expf(-0.5f * t1 * t1) + 0.681f * expf(-0.5f * t2 * t2);
}


//-------------------------------------------------------------------------------------------------
// XYZ to RGB
//

template <typename T>
VSNRAY_FUNC
inline vector<3, T> xyz_to_rgb(vector<3, T> const& xyz)
{
    // see: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    // Assume sRGB working space and D65 reference white
    return matrix<3, 3, T>(
             3.2404542, -0.9692660,  0.0556434,
            -1.5371385,  1.8760108, -0.2040259,
            -0.4985314,  0.0415560,  1.0572252
            )
        * xyz;

}


//-------------------------------------------------------------------------------------------------
// Reflective spectrum to RGB. Requires normalization and mult. with a standard illuminant.
// See: http://www.brucelindbloom.com/index.html?Eqn_Spect_to_XYZ.html
//

template <typename T>
VSNRAY_FUNC
inline vector<3, T> reflective_spd_to_rgb(spectrum<T> const& spe)
{
    const float lmin = spectrum<T>::lambda_min;
    const float lmax = spectrum<T>::lambda_max;
    const float step = (lmax - lmin) / (spectrum<T>::num_samples - 1);

    auto ill = spd_d65();

    T x(0.0);
    T y(0.0);
    T z(0.0);
    T n(0.0);

    for (float lambda = lmin; lambda <= lmax; lambda += step)
    {
        auto p = spe(lambda);
        auto i = ill(lambda);


        x += p * i * cie_x(lambda);
        y += p * i * cie_y(lambda);
        z += p * i * cie_z(lambda);
        n +=     i * cie_y(lambda);
    }

    return xyz_to_rgb( vector<3, T>( x / n, y / n, z / n ) );
}


//-------------------------------------------------------------------------------------------------
// Convert spectrum to luminance (cd/m^2).
// Multiplication with a standard illuminant.
// See: http://www.brucelindbloom.com/index.html?Eqn_Spect_to_XYZ.html
//

template <typename T>
VSNRAY_FUNC
inline T reflective_spd_to_luminance(spectrum<T> const& spe)
{
    const float lmin = spectrum<T>::lambda_min;
    const float lmax = spectrum<T>::lambda_max;
    const float step = (lmax - lmin) / (spectrum<T>::num_samples - 1);

    auto ill = spd_d65();

    T y(0.0);

    for (float lambda = lmin; lambda <= lmax; lambda += step)
    {
        auto p = spe(lambda);
        auto i = ill(lambda);

        y += p * i * cie_y(lambda);
    }

    return y;
}


//-------------------------------------------------------------------------------------------------
// Convert emissive spectrum to RGB
//

template <typename T>
VSNRAY_FUNC
inline vector<3, T> emissive_spd_to_rgb(spectrum<T> const& spe)
{
    const float lmin = spectrum<T>::lambda_min;
    const float lmax = spectrum<T>::lambda_max;
    const float step = (lmax - lmin) / (spectrum<T>::num_samples - 1);

    T x(0.0);
    T y(0.0);
    T z(0.0);
    T n(0.0);

    for (float lambda = lmin; lambda <= lmax; lambda += step)
    {
        auto p = spe(lambda);


        x += p * cie_x(lambda);
        y += p * cie_y(lambda);
        z += p * cie_z(lambda);
        n +=     cie_y(lambda);
    }

    return xyz_to_rgb( vector<3, T>( x / n, y / n, z / n ) );
}


//-------------------------------------------------------------------------------------------------
// Convert emissive spectrum to luminance (cd/m^2)
//

template <typename T>
VSNRAY_FUNC
inline T emissive_spd_to_luminance(spectrum<T> const& spe)
{
    const float lmin = spectrum<T>::lambda_min;
    const float lmax = spectrum<T>::lambda_max;
    const float step = (lmax - lmin) / (spectrum<T>::num_samples - 1);

    T y(0.0);

    for (float lambda = lmin; lambda <= lmax; lambda += step)
    {
        auto p = spe(lambda);

        y += p * cie_y(lambda);
    }

    return y;
}


//-------------------------------------------------------------------------------------------------
// Convert OpenGL pixel formats
//

template <typename TargetType, typename SourceType>
VSNRAY_FUNC
inline void convert(TargetType& target, SourceType const& source)
{
    target = static_cast<TargetType>(source);
}


// RGBAX <-- RGBA32F
template <unsigned Bits>
VSNRAY_FUNC
inline void convert(vector<4, unorm<Bits>>& target, vec4 const& source)
{
    target = vector<4, unorm<Bits>>(clamp(source, vec4(0.0), vec4(1.0)));
}

// RGBA32F <-- RGBA8
// Ok.


} // visionaray


#endif // VSNRAY_DETAIL_COLOR_CONVERSION_H
