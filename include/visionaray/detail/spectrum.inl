// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>
#include <ostream>

#include <visionaray/texture/texture.h>

#include "color_conversion.h"

namespace visionaray
{


//--------------------------------------------------------------------------------------------------
// spectrum members
//

template <typename T>
VSNRAY_FUNC
inline spectrum<T>::spectrum(T c)
    : samples_(c)
{
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>::spectrum(vector<num_samples, T> const& samples)
    : samples_(samples)
{
}

template <typename T>
template <typename U>
VSNRAY_FUNC
inline spectrum<T>::spectrum(spectrum<U> const& rhs)
    : samples_(rhs.samples())
{
}

template <typename T>
template <typename U>
VSNRAY_FUNC
inline spectrum<T>& spectrum<T>::operator=(spectrum<U> const& rhs)
{
    samples_ = rhs.samples_;
    return *this;
}

template <typename T>
VSNRAY_FUNC
inline T& spectrum<T>::operator[](size_t i)
{
    return samples_[i];
}

template <typename T>
VSNRAY_FUNC
inline T const& spectrum<T>::operator[](size_t i) const
{
    return samples_[i];
}

template <typename T>
VSNRAY_FUNC
inline T spectrum<T>::operator()(float lambda) const
{
#if VSNRAY_SPECTRUM_RGB
    float lambda_min = 0.0f;
    float lambda_max = 2.0f;
#endif

    if (lambda < lambda_min || lambda > lambda_max)
    {
        return T(0.0);
    }

    texture_ref<T, ElementType, 1> tex(num_samples);
    tex.set_data( samples_.data() );
    tex.set_filter_mode( Linear );

    float coord = (lambda - lambda_min) / (lambda_max - lambda_min);
    return tex1D(tex, coord);
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator-(spectrum<T> const& s)
{
    return spectrum<T>( -s.samples() );
}

// spectrum op spectrum

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator+(spectrum<T> const& s, spectrum<T> const& t)
{
    return spectrum<T>( s.samples() + t.samples() );
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator-(spectrum<T> const& s, spectrum<T> const& t)
{
    return spectrum<T>( s.samples() - t.samples() );
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator*(spectrum<T> const& s, spectrum<T> const& t)
{
    return spectrum<T>( s.samples() * t.samples() );
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator/(spectrum<T> const& s, spectrum<T> const& t)
{
    return spectrum<T>( s.samples() / t.samples() );
}

// spectrum op scalar

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator+(spectrum<T> const& s, T t)
{
    return spectrum<T>( s.samples() + t );
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator-(spectrum<T> const& s, T t)
{
    return spectrum<T>( s.samples() - t );
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator*(spectrum<T> const& s, T t)
{
    return spectrum<T>( s.samples() * t );
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator/(spectrum<T> const& s, T t)
{
    return spectrum<T>( s.samples() / t );
}

// scalar op spectrum

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator+(T s, spectrum<T> const& t)
{
    return spectrum<T>( s + t.samples() );
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator-(T s, spectrum<T> const& t)
{
    return spectrum<T>( s - t.samples() );
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator*(T s, spectrum<T> const& t)
{
    return spectrum<T>( s * t.samples() );
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> operator/(T s, spectrum<T> const& t)
{
    return spectrum<T>( s / t.samples() );
}

// append operations

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& operator+=(spectrum<T>& s, spectrum<T> const& t)
{
    s = spectrum<T>(s + t);
    return s;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& operator-=(spectrum<T>& s, spectrum<T> const& t)
{
    s = spectrum<T>(s - t);
    return s;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& operator*=(spectrum<T>& s, spectrum<T> const& t)
{
    s = spectrum<T>(s * t);
    return s;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& operator/=(spectrum<T>& s, spectrum<T> const& t)
{
    s = spectrum<T>(s / t);
    return s;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& operator+=(spectrum<T>& s, T t)
{
    s = spectrum<T>(s + t);
    return s;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& operator-=(spectrum<T>& s, T t)
{
    s = spectrum<T>(s - t);
    return s;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& operator*=(spectrum<T>& s, T t)
{
    s = spectrum<T>(s * t);
    return s;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& operator/=(spectrum<T>& s, T t)
{
    s = spectrum<T>(s / t);
    return s;
}


//-------------------------------------------------------------------------------------------------
// Misc.
//

template <typename M, typename T>
VSNRAY_FUNC
inline spectrum<T> select(M const& m, spectrum<T> const& s, spectrum<T> const& t)
{
    return spectrum<T>( select(m, s.samples(), t.samples()) );
}

template <typename T>
VSNRAY_FUNC
inline T mean_value(spectrum<T> const& s)
{
    return hadd( s.samples() ) / T(spectrum<T>::num_samples);
}

template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, spectrum<T> const& v)
{
    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(';
    for (size_t d = 0; d < spectrum<T>::num_samples; ++d)
    {
        s << v[d];
        if (d < spectrum<T>::num_samples - 1)
        {
            s << ',';
        }
    }
    s << ')';

    return out << s.str();
}


//-------------------------------------------------------------------------------------------------
// Conversions
//

// SPD -> SPD

template <typename T, typename SPD>
VSNRAY_FUNC
inline spectrum<T> from_spd(SPD spd)
{
#if VSNRAY_SPECTRUM_RGB
    float lambda_min = 0.0f;
    float lambda_max = 2.0f;
#else
    float lambda_min = spectrum<T>::lambda_min;
    float lambda_max = spectrum<T>::lambda_max;
#endif

    spectrum<T> result;

    for (size_t i = 0; i < spectrum<T>::num_samples; ++i)
    {
        float f = i / static_cast<float>(spectrum<T>::num_samples - 1);
        float lambda = lerp( lambda_min, lambda_max, f );
        result.samples()[i] = spd(lambda);
    }

    return result;
}

// RGB -> SPD

template <typename T>
VSNRAY_FUNC
inline spectrum<T> from_rgb(vector<3, T> const& rgb)
{
#if VSNRAY_SPECTRUM_RGB
    return spectrum<T>(rgb);
#else

    // TODO: http://www.cs.utah.edu/~bes/papers/color/paper.pdf

    spectrum<T> result;

    for (size_t i = 0; i < spectrum<T>::num_samples; ++i)
    {
        auto bin = (i * 3) / spectrum<T>::num_samples;

        if (bin == 0)
        {
            result[i] = rgb.z;
        }
        else if (bin == 1)
        {
            result[i] = rgb.y;
        }
        else
        {
            result[i] = rgb.x;
        }
    }

    return result;

#endif
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> from_rgb(T r, T g, T b)
{
    return from_rgb( vector<3, T>(r, g, b) );
}

// RGBA -> SPD

template <typename T>
VSNRAY_FUNC
inline spectrum<T> from_rgba(vector<4, T> const& rgba)
{
    return from_rgb( rgba.x * rgba.w, rgba.y * rgba.w, rgba.z * rgba.w );
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> from_rgba(T r, T g, T b, T a)
{
    return from_rgba(r, g, b, a);
}

// SPD -> Luminance

template <typename T>
VSNRAY_FUNC
inline T to_luminance(spectrum<T> const& s)
{
#if VSNRAY_SPECTRUM_RGB
    return T(0.3) * s[0] + T(0.59) * s[1] + T(0.11) * s[2];
#else
    return spd_to_luminance(s);
#endif
}

// SPD -> RGB

template <typename T>
VSNRAY_FUNC
inline vector<3, T> to_rgb(spectrum<T> const& s)
{
#if VSNRAY_SPECTRUM_RGB
    static_assert( spectrum<T>::num_samples == 3, "Incompatible num samples" );
    return s.samples();
#else
    static_assert( spectrum<T>::num_samples > 1, "Incompatible num samples" );
    return spd_to_rgb(s);
#endif
}

// SPD -> RGBA

template <typename T>
VSNRAY_FUNC
inline vector<4, T> to_rgba(spectrum<T> const& s)
{
    return vector<4, T>( to_rgb(s), T(1.0) );
}


namespace simd
{

//-------------------------------------------------------------------------------------------------
// SIMD conversions
//

template <size_t N>
inline spectrum<typename float_from_simd_width<N>::type> pack(
        std::array<spectrum<float>, N> const& specs
        )
{
    using T = typename float_from_simd_width<N>::type;
    using V = vector<spectrum<T>::num_samples, float>;

    std::array<V, N> arr;

    for (size_t i = 0; i < N; ++i)
    {
        arr[i] = specs[i].samples();
    }

    return spectrum<T>(pack(arr));
}

} // simd

} // visionaray
