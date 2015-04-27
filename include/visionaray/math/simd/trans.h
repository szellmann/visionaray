// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_SIMD_TRANS_H
#define VSNRAY_MATH_SIMD_TRANS_H

//-------------------------------------------------------------------------------------------------
// minimax polynomial approximations for transcendental functions
// cf. David H. Eberly: GPGPU Programming for Games and Science, pp. 120
//


#include "sse.h"
#include "../detail/math.h"


namespace MATH_NAMESPACE
{
namespace simd
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// detail::frexp and detail::scalbn do not handle subnormals!
//

template <typename F, typename I>
inline basic_float<F> frexp(basic_float<F> const& x, basic_int<I>* exp)
{
    typedef basic_int<I>        int_type;
    typedef basic_mask<F, I>    mask_type;

    static const int_type exp_mask(0x7f800000);
    static const int_type inv_exp_mask(~0x7f800000);
    static const int_type man_mask(0x3f000000);

    int_type ptr = reinterpret_as_int(x);
    *exp = (ptr & exp_mask) >> 23;
    mask_type is_zero = (*exp == 0);
    *exp = select( is_zero, int_type(0), *exp - 126 ); // IEEE-754 stores a biased exponent
    ptr  = select( is_zero, int_type(0), ptr & inv_exp_mask );
    ptr  = select( is_zero, int_type(0), ptr | man_mask );
    return reinterpret_as_float(ptr);
}

template <typename F, typename I>
inline basic_float<F> scalbn(basic_float<F> const& x, basic_int<I> const& exp)
{
    typedef basic_int<I>        int_type;
    typedef basic_float<F>      float_type;
    typedef basic_mask<F, I>    mask_type;

    static const int_type exp_mask(0x7f800000);
    static const float_type huge_val = reinterpret_as_float(int_type(0x7f800000));
    static const float_type tiny_val = reinterpret_as_float(int_type(0x00000000));

    int_type xi = reinterpret_as_int(x);
    float_type sign = reinterpret_as_float(xi & 0x80000000);
    int_type k = (xi & exp_mask) >> 23;
    k += exp;

    // overflow?
    mask_type uoflow = k > int_type(0xfe);
    float_type huge_or_tiny = select(uoflow, huge_val, tiny_val) | sign;

    // overflow or underflow?
    uoflow |= k < int_type(0);
    return select( uoflow, huge_or_tiny, reinterpret_as_float((xi & int_type(0x807fffff)) | (k << 23)) );
}


//-------------------------------------------------------------------------------------------------
// Polynomials with degree D
//

template <unsigned D>
struct poly_t
{

    template <typename T>
    static T eval(T x, T const* p)
    {

        T result(0.0);
        T y(1.0);

        for (unsigned i = 0; i <= D; ++i)
        {
            result += p[i] * y;
            y *= x;
        }

        return result;

    }
};

template <unsigned D>
struct pow2_t;

template <>
struct pow2_t<1> : public poly_t<1>
{
    template <typename T>
    static T value(T x)
    {
        static const T p[] =
        {
            T(1.0), T(1.0)
        };

        return poly_t::eval(x, p);
    }
};

template <>
struct pow2_t<2> : public poly_t<2>
{
    template <typename T>
    static T value(T x)
    {
        static const T p[] =
        {
            T(1.0),
            T(6.5571332605741528E-1),
            T(3.4428667394258472E-1)
        };

        return poly_t::eval(x, p);
    }
};

template <>
struct pow2_t<3> : public poly_t<3>
{
    template <typename T>
    static T value(T x)
    {
        static const T p[] =
        {
            T(1.0),
            T(6.9589012084456225E-1),
            T(2.2486494900110188E-1),
            T(7.9244930154334980E-2)
        };

        return poly_t::eval(x, p);
    }
};

template <>
struct pow2_t<4> : public poly_t<4>
{
    template <typename T>
    static T value(T x)
    {
        static const T p[] =
        {
            T(1.0),
            T(6.9300392358459195E-1),
            T(2.4154981722455560E-1),
            T(5.1744260331489045E-2),
            T(1.3701998859367848E-2)
        };

        return poly_t::eval(x, p);
    }
};

template <>
struct pow2_t<5> : public poly_t<5>
{
    template <typename T>
    static T value(T x)
    {
        static const T p[] =
        {
            T(1.0),
            T(6.9315298010274962E-1),
            T(2.4014712313022102E-1),
            T(5.5855296413199085E-2),
            T(8.9477503096873079E-3),
            T(1.8968500441332026E-3)
        };

        return poly_t::eval(x, p);
    }
};

template <>
struct pow2_t<6> : public poly_t<6>
{
    template <typename T>
    static T value(T x)
    {
        static const T p[] =
        {
            T(1.0),
            T(6.9314698914837525E-1),
            T(2.4023013440952923E-1),
            T(5.5481276898206033E-2),
            T(9.6838443037086108E-3),
            T(1.2388324048515642E-3),
            T(2.1892283501756538E-4)
        };

        return poly_t::eval(x, p);
    }
};

template <>
struct pow2_t<7> : public poly_t<7>
{
    template <typename T>
    static T value(T x)
    {
        static const T p[] =
        {
            T(1.0),
            T(6.9314718588750690E-1),
            T(2.4022637363165700E-1),
            T(5.5505235570535660E-2),
            T(9.6136265387940512E-3),
            T(1.3429234504656051E-3),
            T(1.4299202757683815E-4),
            T(2.1662892777385423E-5)
        };

        return poly_t::eval(x, p);
    }
};


inline float4 pow2(float4 x)
{
    float4 xi = floor(x);
    float4 xf = x - xi;
    return detail::scalbn(float4(1.0), int4(xi)) * pow2_t<7>::value(xf);
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

inline float8 pow2(float8 x)
{
    float8 xi = floor(x);
    float8 xf = x - xi;
    return detail::scalbn(float8(1.0), int8(xi)) * pow2_t<7>::value(xf);
}

#endif

//-------------------------------------------------------------------------------------------------
// log2(1 + x), x in [0,1)
//

template <unsigned D>
struct log2_t;

template <>
struct log2_t<1> : public poly_t<1>
{
    template <typename T>
    static T value(T x)
    {
        static const T p[] =
        {
            T(0.0), T(1.0)
        };

        return poly_t::eval(x, p);
    }
};

template <>
struct log2_t<7> : public poly_t<7>
{
    template <typename T>
    static T value(T x)
    {
        static const T p[] =
        {
            T(0.0),
            T(+1.4426664401536078),
            T(-7.2055423726162360E-1),
            T(+4.7332419162501083E-1),
            T(-3.2514018752954144E-1),
            T(+1.9302966529095673E-1),
            T(-7.8534970641157997E-2),
            T(+1.5209108363023915E-2)
        };

        return poly_t::eval(x, p);
    }
};

template <>
struct log2_t<8> : public poly_t<8>
{
    template <typename T>
    static T value(T x)
    {
        static const T p[] =
        {
            T(0.0),
            T(+1.4426896453621882),
            T(-7.2115893912535967E-1),
            T(+4.7861716616785088E-1),
            T(-3.4699935395019565E-1),
            T(+2.4114048765477492E-1),
            T(-1.3657398692885181E-1),
            T(+5.1421382871922106E-2),
            T(-9.1364020499895560E-3)
        };

        return poly_t::eval(x, p);
    }
};


template <typename T>
inline T log2(T x)
{
    return log2_t<8>::value(x);
}

} // detail


//-------------------------------------------------------------------------------------------------
// API
//

float4 exp(float4 x);
float4 log(float4 x);
float4 log2(float4 x);
float4 pow(float4 x, float4 y);

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
float8 exp(float8 x);
float8 log(float8 x);
float8 log2(float8 x);
float8 pow(float8 x, float8 y);
#endif


//-------------------------------------------------------------------------------------------------
// impl
//

inline float4 cos(float4 x)
{
    VSNRAY_ALIGN(16) float tmp[4];
    store(tmp, x);

    return float4( std::cos(tmp[0]), std::cos(tmp[1]), std::cos(tmp[2]), std::cos(tmp[3]) );
}

inline float4 sin(float4 x)
{
    VSNRAY_ALIGN(16) float tmp[4];
    store(tmp, x);

    return float4( std::sin(tmp[0]), std::sin(tmp[1]), std::sin(tmp[2]), std::sin(tmp[3]) );
}

inline float4 exp(float4 x)
{
    float4 y = x * constants::log2_e<float4>();
    return detail::pow2(y);
}

inline float4 log(float4 x)
{
    return log2(x) / constants::log2_e<float4>();
}

inline float4 log2(float4 x)
{
    int4 n = 0;
    float4 m = detail::frexp(x, &n);
    m *= 2.0f;
    return float4(n - 1) + detail::log2(m - 1.0f);
}

inline float4 pow(float4 x, float4 y)
{
#if VSNRAY_SIMD_HAS_SVML
    return _mm_pow_ps(x, y);
#else
    return exp( y * log(x) );
#endif
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// TODO: consolidate stuff with float4 (template)

inline float8 cos(float8 x)
{
    VSNRAY_ALIGN(32) float tmp[8];
    store(tmp, x);

    return float8
    (
        std::cos(tmp[0]), std::cos(tmp[1]), std::cos(tmp[2]), std::cos(tmp[3]),
        std::cos(tmp[4]), std::cos(tmp[5]), std::cos(tmp[6]), std::cos(tmp[7])
    );
}

inline float8 sin(float8 x)
{
    VSNRAY_ALIGN(32) float tmp[8];
    store(tmp, x);

    return float8
    (
        std::sin(tmp[0]), std::sin(tmp[1]), std::sin(tmp[2]), std::sin(tmp[3]),
        std::sin(tmp[4]), std::sin(tmp[5]), std::sin(tmp[6]), std::sin(tmp[7])
    );
}

inline float8 exp(float8 x)
{
    float8 y = x * constants::log2_e<float8>();
    return detail::pow2(y);
}

inline float8 log(float8 x)
{
    return log2(x) / constants::log2_e<float8>();
}

inline float8 log2(float8 x)
{
    int8 n = 0;
    float8 m = detail::frexp(x, &n);
    m *= 2.0f;
    return float8(n - 1) + detail::log2(m - 1.0f);
}

inline float8 pow(float8 x, float8 y)
{
#if VSNRAY_SIMD_HAS_SVML
    return _mm256_pow_ps(x, y);
#else
    return exp( y * log(x) );
#endif
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

} // simd
} // MATH_NAMESPACE

#endif // VSNRAY_MATH_SIMD_TRANS_H
