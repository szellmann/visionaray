// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SPECTRUM_H
#define VSNRAY_SPECTRUM_H 1

#include <cstddef>

#include "detail/macros.h"
#include "math/vector.h"


//-------------------------------------------------------------------------------------------------
// If set to 0, Visionaray stores sampled spectral power distributions in class spectrum
//

#define VSNRAY_SPECTRUM_RGB 1

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Spectral power distribution
//

template <typename T>
class spectrum
{
public:

#if VSNRAY_SPECTRUM_RGB
    enum { num_samples = 3 };
#else
    enum { num_samples = 300 };
    static constexpr float  lambda_min  = 400.0f;
    static constexpr float  lambda_max  = 700.0f;
#endif

public:

    spectrum() = default;

    VSNRAY_FUNC explicit spectrum(T const& c);
    VSNRAY_FUNC explicit spectrum(vector<num_samples, T> const& samples);

    template <typename U>
    VSNRAY_FUNC explicit spectrum(spectrum<U> const& rhs);

    template <typename U>
    VSNRAY_FUNC spectrum& operator=(spectrum<U> const& rhs);

    VSNRAY_FUNC T& operator[](size_t i);
    VSNRAY_FUNC T const& operator[](size_t i) const;

    VSNRAY_FUNC T operator()(float lambda) const;

    VSNRAY_FUNC vector<num_samples, T>&       samples()       { return samples_; }
    VSNRAY_FUNC vector<num_samples, T> const& samples() const { return samples_; }

private:

    vector<num_samples, T> samples_;

};

} // visionaray

#include "detail/spectrum.inl"

#endif // VSNRAY_SPECTRUM_H
