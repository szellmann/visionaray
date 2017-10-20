// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_SIMD_TRANS_H
#define VSNRAY_MATH_SIMD_TRANS_H 1

//-------------------------------------------------------------------------------------------------
// minimax polynomial approximations for transcendental functions
// cf. David H. Eberly: GPGPU Programming for Games and Science, pp. 120
//

#include <type_traits>

#include "../detail/math.h"
#include "../constants.h"
#include "avx.h"
#include "avx512.h"
#include "builtin.h"
#include "neon.h"
#include "sse.h"
#include "type_traits.h"


namespace MATH_NAMESPACE
{
namespace simd
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// detail::frexp and detail::scalbn do not handle subnormals!
//

template <
    typename F,
    typename I,
    typename = typename std::enable_if<is_simd_vector<F>::value>::type,
    typename = typename std::enable_if<is_simd_vector<I>::value>::type
    >
MATH_FUNC
VSNRAY_FORCE_INLINE F frexp(F const& x, I* exp)
{
    using M = mask_type_t<F>;

    static const I exp_mask(0x7f800000);
    static const I inv_exp_mask(~0x7f800000);
    static const I man_mask(0x3f000000);

    I ptr = reinterpret_as_int(x);
    *exp = (ptr & exp_mask) >> 23;
    M is_zero = (*exp == 0);
    *exp = select( is_zero, I(0), *exp - 126 ); // IEEE-754 stores a biased exponent
    ptr  = select( is_zero, I(0), ptr & inv_exp_mask );
    ptr  = select( is_zero, I(0), ptr | man_mask );
    return reinterpret_as_float(ptr);
}

template <
    typename F,
    typename I,
    typename = typename std::enable_if<is_simd_vector<F>::value>::type,
    typename = typename std::enable_if<is_simd_vector<I>::value>::type
    >
MATH_FUNC
VSNRAY_FORCE_INLINE F scalbn(F const& x, I const& exp)
{
    using M = mask_type_t<F>;

    static const I exp_mask(0x7f800000);
    static const F huge_val = reinterpret_as_float(I(0x7f800000));
    static const F tiny_val = reinterpret_as_float(I(0x00000000));

    I xi = reinterpret_as_int(x);
    F sign = reinterpret_as_float(xi & 0x80000000);
    I k = (xi & exp_mask) >> 23;
    k += exp;

    // overflow?
    M uoflow = k > I(0xfe);
    F huge_or_tiny = select(uoflow, huge_val, tiny_val) | sign;

    // overflow or underflow?
    uoflow |= k < I(0);
    return select( uoflow, huge_or_tiny, reinterpret_as_float((xi & I(0x807fffff)) | (k << 23)) );
}


//-------------------------------------------------------------------------------------------------
// Polynomials with degree D
//

template <unsigned D>
struct poly_t
{

    template <typename T>
    MATH_FUNC
    static T eval(T const& x, T const* p)
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
    MATH_FUNC
    static T value(T const& x)
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
    MATH_FUNC
    static T value(T const& x)
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
    MATH_FUNC
    static T value(T const& x)
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
    MATH_FUNC
    static T value(T const& x)
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
    MATH_FUNC
    static T value(T const& x)
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
    MATH_FUNC
    static T value(T const& x)
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
    MATH_FUNC
    static T value(T const& x)
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


template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
MATH_FUNC
VSNRAY_FORCE_INLINE FloatT pow2(FloatT const& x)
{
    FloatT xi = floor(x);
    FloatT xf = x - xi;
    return detail::scalbn(FloatT(1.0), convert_to_int(xi)) * pow2_t<7>::value(xf);
}


//-------------------------------------------------------------------------------------------------
// log2(1 + x), x in [0,1)
//

template <unsigned D>
struct log2_t;

template <>
struct log2_t<1> : public poly_t<1>
{
    template <typename T>
    MATH_FUNC
    static T value(T const& x)
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
    MATH_FUNC
    static T value(T const& x)
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
    MATH_FUNC
    static T value(T const& x)
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
MATH_FUNC
VSNRAY_FORCE_INLINE T log2(T const& x)
{
    return log2_t<8>::value(x);
}

} // detail


//-------------------------------------------------------------------------------------------------
// Trigonometric functions
// TODO: implement w/o context switch
//

template <typename F, typename = typename std::enable_if<is_simd_vector<F>::value>::type>
MATH_FUNC
VSNRAY_FORCE_INLINE F cos(F const& x)
{
    using float_array = aligned_array_t<F>;

    float_array tmp;
    store(tmp, x);

    for (size_t i = 0; i < num_elements<F>::value; ++i)
    {
        tmp[i] = cosf(tmp[i]);
    }

    return F(tmp);
}

template <typename F, typename = typename std::enable_if<is_simd_vector<F>::value>::type>
MATH_FUNC
VSNRAY_FORCE_INLINE F sin(F const& x)
{
    using float_array = aligned_array_t<F>;

    float_array tmp;
    store(tmp, x);

    for (size_t i = 0; i < num_elements<F>::value; ++i)
    {
        tmp[i] = sinf(tmp[i]);
    }

    return F(tmp);
}

template <typename F, typename = typename std::enable_if<is_simd_vector<F>::value>::type>
MATH_FUNC
VSNRAY_FORCE_INLINE F tan(F const& x)
{
    using float_array = aligned_array_t<F>;

    float_array tmp;
    store(tmp, x);

    for (size_t i = 0; i < num_elements<F>::value; ++i)
    {
        tmp[i] = tanf(tmp[i]);
    }

    return F(tmp);
}

template <typename F, typename = typename std::enable_if<is_simd_vector<F>::value>::type>
MATH_FUNC
VSNRAY_FORCE_INLINE F acos(F const& x)
{
    using float_array = aligned_array_t<F>;

    float_array tmp;
    store(tmp, x);

    for (size_t i = 0; i < num_elements<F>::value; ++i)
    {
        tmp[i] = acosf(tmp[i]);
    }

    return F(tmp);
}

template <typename F, typename = typename std::enable_if<is_simd_vector<F>::value>::type>
MATH_FUNC
VSNRAY_FORCE_INLINE F asin(F const& x)
{
    using float_array = aligned_array_t<F>;

    float_array tmp;
    store(tmp, x);

    for (size_t i = 0; i < num_elements<F>::value; ++i)
    {
        tmp[i] = asinf(tmp[i]);
    }

    return F(tmp);
}

template <typename F, typename = typename std::enable_if<is_simd_vector<F>::value>::type>
MATH_FUNC
VSNRAY_FORCE_INLINE F atan(F const& x)
{
    using float_array = aligned_array_t<F>;

    float_array tmp;
    store(tmp, x);

    for (size_t i = 0; i < num_elements<F>::value; ++i)
    {
        tmp[i] = atanf(tmp[i]);
    }

    return F(tmp);
}


//-------------------------------------------------------------------------------------------------
// exp() / log() / log2()
//

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
MATH_FUNC
VSNRAY_FORCE_INLINE FloatT exp(FloatT const& x)
{
    FloatT y = x * constants::log2_e<FloatT>();
    return detail::pow2(y);
}

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
MATH_FUNC
VSNRAY_FORCE_INLINE FloatT log(FloatT const& x)
{
    return log2(x) / constants::log2_e<FloatT>();
}

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
MATH_FUNC
VSNRAY_FORCE_INLINE FloatT log2(FloatT const& x)
{
    using IntT = int_type_t<FloatT>;

    IntT n = 0;
    FloatT m = detail::frexp(x, &n);
    m *= 2.0f;
    return convert_to_float(n - 1) + detail::log2(m - 1.0f);
}


//-------------------------------------------------------------------------------------------------
// pow()
//

MATH_FUNC
VSNRAY_FORCE_INLINE float4 pow(float4 const& x, float4 const& y)
{
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE2) && VSNRAY_SIMD_HAS_SVML
    return _mm_pow_ps(x, y);
#elif VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE2) || VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_NEON_FP)
    return exp( y * log(x) );
#else
    // No dedicated simd instructions
    return float4(
            powf(x.value[0], y.value[0]),
            powf(x.value[1], y.value[1]),
            powf(x.value[2], y.value[2]),
            powf(x.value[3], y.value[3])
            );
#endif
}

VSNRAY_FORCE_INLINE float8 pow(float8 const& x, float8 const& y)
{
#if VSNRAY_SIMD_HAS_SVML
    return _mm256_pow_ps(x, y);
#elif VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    return exp( y * log(x) );
#else
    // No dedicated simd instructions
    return float8(
            powf(x.value[0], y.value[0]),
            powf(x.value[1], y.value[1]),
            powf(x.value[2], y.value[2]),
            powf(x.value[3], y.value[3]),
            powf(x.value[4], y.value[4]),
            powf(x.value[5], y.value[5]),
            powf(x.value[6], y.value[6]),
            powf(x.value[7], y.value[7])
            );
#endif
}

VSNRAY_FORCE_INLINE float16 pow(float16 const& x, float16 const& y)
{
#if VSNRAY_SIMD_HAS_SVML
    return _mm512_pow_ps(x, y); // TODO: exists?
#elif VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
    return exp( y * log(x) );
#else
    // No dedicated simd instructions
    return float16(
            powf(x.value[ 0], y.value[ 0]),
            powf(x.value[ 1], y.value[ 1]),
            powf(x.value[ 2], y.value[ 2]),
            powf(x.value[ 3], y.value[ 3]),
            powf(x.value[ 4], y.value[ 4]),
            powf(x.value[ 5], y.value[ 5]),
            powf(x.value[ 6], y.value[ 6]),
            powf(x.value[ 7], y.value[ 7]),
            powf(x.value[ 8], y.value[ 8]),
            powf(x.value[ 9], y.value[ 9]),
            powf(x.value[10], y.value[10]),
            powf(x.value[11], y.value[11]),
            powf(x.value[12], y.value[12]),
            powf(x.value[13], y.value[13]),
            powf(x.value[14], y.value[14]),
            powf(x.value[15], y.value[15])
            );
#endif
}

} // simd
} // MATH_NAMESPACE

#endif // VSNRAY_MATH_SIMD_TRANS_H
