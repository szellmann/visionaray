// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_COLOR_CONVERSION_H
#define VSNRAY_DETAIL_COLOR_CONVERSION_H 1

#include <cassert>
#include <cstddef>

#include <visionaray/math/detail/math.h>
#include <visionaray/math/matrix.h>
#include <visionaray/math/unorm.h>
#include <visionaray/math/vector.h>
#include <visionaray/pixel_format.h>
#include <visionaray/spectrum.h>

#include "spd/blackbody.h"
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

    return 0.362f * exp(-0.5f * t1 * t1) + 1.056f * exp(-0.5f * t2 * t2) - 0.065f * exp(-0.5f * t3 * t3);
}

VSNRAY_FUNC
inline float cie_y(float lambda)
{
    float t1 = (lambda - 568.8f) * ((lambda < 568.8f) ? 0.0213f : 0.0247f);
    float t2 = (lambda - 530.9f) * ((lambda < 530.9f) ? 0.0613f : 0.0322f);

    return 0.821f * exp(-0.5f * t1 * t1) + 0.286f * exp(-0.5f * t2 * t2);
}

VSNRAY_FUNC
inline float cie_z(float lambda)
{
    float t1 = (lambda - 437.0f) * ((lambda < 437.0f) ? 0.0845f : 0.0278f);
    float t2 = (lambda - 459.0f) * ((lambda < 459.0f) ? 0.0385f : 0.0725f);

    return 1.217f * exp(-0.5f * t1 * t1) + 0.681f * exp(-0.5f * t2 * t2);
}


//-------------------------------------------------------------------------------------------------
// Hue and temperature to RGB
//

template <typename T>
VSNRAY_FUNC
inline vector<3, T> hue_to_rgb(T hue)
{
//  assert(hue >= 0.0f && hue <= 1.0f);

    T s = saturate( hue ) * T(6.0f);

    T r = saturate( abs(s - T(3.0)) - T(1.0) );
    T g = saturate( T(2.0) - abs(s - T(2.0)) );
    T b = saturate( T(2.0) - abs(s - T(4.0)) );

    return vector<3, T>(r, g, b);
}

template <typename T>
VSNRAY_FUNC
inline vector<3, T> temperature_to_rgb(T t)
{
    T K = T(4.0f / 6.0f);

    T h = K - K * t;
    T v = T(0.5f) + T(0.5f) * t;

    return v * hue_to_rgb(h);
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
// Convert an arbitrary spectral power distribution to RGB
//

template <typename SPD, typename T = float>
VSNRAY_FUNC
inline vector<3, T> spd_to_rgb(SPD const& spd, T lmin = T(400.0), T lmax = T(700.0), T step = T(1.0))
{
    T x(0.0);
    T y(0.0);
    T z(0.0);
    T n(0.0);

    for (float lambda = lmin; lambda <= lmax; lambda += step)
    {
        auto p = spd(lambda);

        x += p * cie_x(lambda);
        y += p * cie_y(lambda);
        z += p * cie_z(lambda);
        n +=     cie_y(lambda);
    }

    return xyz_to_rgb( vector<3, T>( x / n, y / n, z / n ) );
}


//-------------------------------------------------------------------------------------------------
// Convert an arbitrary spectral power distribution to RGB
//

VSNRAY_FUNC
inline vector<3, float> spd_to_rgb(blackbody const& spd, float lmin = 400.0f, float lmax = 700.0f, float step = 1.0f, bool normalize = true)
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float n = 0.0f;

    // For normalization
    float max_radiance = 1.0f;

    if (normalize)
    {
        // lambda where radiance is max
        float lambda_max_radiance = 2.8977721e-3 / spd.temperature() * 1e9 /* m2nm */;

        max_radiance = spd(lambda_max_radiance);
    }

    for (float lambda = lmin; lambda <= lmax; lambda += step)
    {
        auto p = spd(lambda) / max_radiance;

        x += p * cie_x(lambda);
        y += p * cie_y(lambda);
        z += p * cie_z(lambda);
        n +=     cie_y(lambda);
    }

    return xyz_to_rgb( vector<3, float>( x / n, y / n, z / n ) );
}


//-------------------------------------------------------------------------------------------------
// Convert spectrum to RGB
//

template <typename T>
VSNRAY_FUNC
inline vector<3, T> spd_to_rgb(spectrum<T> const& spe)
{
    const float lmin = spectrum<T>::lambda_min;
    const float lmax = spectrum<T>::lambda_max;
    const float step = (lmax - lmin) / (spectrum<T>::num_samples - 1);

    return spd_to_rgb(spe, lmin, lmax, step);
}


//-------------------------------------------------------------------------------------------------
// Convert RGB to luminance (cd/m^2)
//

template <typename T>
VSNRAY_FUNC
inline T rgb_to_luminance(vector<3, T> const& rgb)
{
    return T(0.3) * rgb.x + T(0.59) * rgb.y + T(0.11) * rgb.z;
}


//-------------------------------------------------------------------------------------------------
// Convert spectrum to luminance (cd/m^2)
//

template <typename T>
VSNRAY_FUNC
inline T spd_to_luminance(spectrum<T> const& spe)
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
// Convert from linear to sRGB space
//

template <typename T>
VSNRAY_FUNC
inline vector<3, T> linear_to_srgb(vector<3, T> const& rgb)
{
    vector<3, T> result;

    for (int i = 0; i < 3; ++i)
    {
        result[i] = select(
            rgb[i] <= T(0.0031308),
            T(12.92) * rgb[i],
            T(1.055) * pow(rgb[i], T(1.0 / 2.4)) - T(0.055)
            );
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Convert from sRGB to linear space
//

template <typename T>
VSNRAY_FUNC
inline vector<3, T> srgb_to_linear(vector<3, T> const& srgb)
{
    vector<3, T> result;

    for (int i = 0; i < 3; ++i)
    {
        result[i] = select(
            srgb[i] <= T(0.04045),
            srgb[i] / T(12.92),
            pow((srgb[i] + T(0.055)) / T(1.055), T(2.4))
            );
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Convert OpenGL pixel formats
//

template <pixel_format TF, pixel_format SF, typename TargetType, typename SourceType>
VSNRAY_FUNC
inline void convert(
        pixel_format_constant<TF>   /* */,
        pixel_format_constant<SF>   /* */,
        TargetType&                 target,
        SourceType const&           source
        )
{
    target = static_cast<TargetType>(source);
}

// To PF_UNSPECIFIED (noop)
template <pixel_format SF, typename TargetType, typename SourceType>
VSNRAY_FUNC
inline void convert(
        pixel_format_constant<PF_UNSPECIFIED> /* */,
        pixel_format_constant<SF>             /* */,
        TargetType&                           /* */,
        SourceType const&                     /* */
        )
{
    assert(0);
}

// From PF_UNSPECIFIED (noop)
template <pixel_format TF, typename TargetType, typename SourceType>
VSNRAY_FUNC
inline void convert(
        pixel_format_constant<TF>             /* */,
        pixel_format_constant<PF_UNSPECIFIED> /* */,
        TargetType&                           /* */,
        SourceType const&                     /* */
        )
{
    assert(0);
}

// PF_DEPTH32F to PF_DEPTH24_STENCIL8 conversion

template <typename TargetType, typename SourceType>
VSNRAY_FUNC
inline void convert(
        pixel_format_constant<PF_DEPTH24_STENCIL8>  /* */,
        pixel_format_constant<PF_DEPTH32F>          /* */,
        TargetType&                                 target,
        SourceType const&                           source
        )
{
    auto depth_raw = convert_to_int(source * 16777215.0f);
    target = (depth_raw << 8);
}


// PF_DEPTH24_STENCIL8 to PF_DEPTH32F conversion

template <typename TargetType, typename SourceType>
VSNRAY_FUNC
inline void convert(
        pixel_format_constant<PF_DEPTH32F>          /* */,
        pixel_format_constant<PF_DEPTH24_STENCIL8>  /* */,
        TargetType&                                 target,
        SourceType const&                           source
        )
{
    auto depth_raw = (source & 0xFFFFFF00) >> 8;
    target = TargetType(depth_raw) / 16777215.0f;
}


// float to unorm conversion
template <pixel_format TF, pixel_format SF, unsigned Bits, size_t Dim>
VSNRAY_FUNC
inline void convert(
        pixel_format_constant<TF>   /* */,
        pixel_format_constant<SF>   /* */,
        vector<Dim, unorm<Bits>>&   target,
        vector<Dim, float> const&   source
        )
{
    using V = vector<Dim, float>;
    target = vector<Dim, unorm<Bits>>(clamp(source, V(0.0), V(1.0)));
}

// unorm to float conversion
// Ok.


// RGBA to RGB conversion, multiply by alpha
template <pixel_format TF, pixel_format SF, typename T, typename U>
VSNRAY_FUNC
inline void convert(
        pixel_format_constant<TF>   /* */,
        pixel_format_constant<SF>   /* */,
        vector<3, T>&               target,
        vector<4, U> const&         source
        )
{
    convert(
        pixel_format_constant<TF>{},
        pixel_format_constant<SF>{},
        target,
        source.xyz() * source.w
        );
}

// RGB to RGBA conversion, let alpha = 1.0
template <pixel_format TF, pixel_format SF, typename T, typename U>
VSNRAY_FUNC
inline void convert(
        pixel_format_constant<TF>   /* */,
        pixel_format_constant<SF>   /* */,
        vector<4, T>&               target,
        vector<3, U> const&         source
        )
{
    convert(
        pixel_format_constant<TF>{},
        pixel_format_constant<SF>{},
        target,
        vector<4, U>(source, U(1.0))
        );
}

} // visionaray


#endif // VSNRAY_DETAIL_COLOR_CONVERSION_H
