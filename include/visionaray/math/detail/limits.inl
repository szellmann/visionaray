// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cfloat>
#include <climits>
#include <limits>

namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// generic limits: no CUDA
//

template <typename T>
inline T numeric_limits<T>::min()
{
    return std::numeric_limits<T>::min();
}

template <typename T>
inline T numeric_limits<T>::lowest()
{
    return std::numeric_limits<T>::lowest();
}

template <typename T>
inline T numeric_limits<T>::max()
{
    return std::numeric_limits<T>::max();
}

template <typename T>
inline T numeric_limits<T>::epsilon()
{
    return std::numeric_limits<T>::epsilon();
}


//-------------------------------------------------------------------------------------------------
// int limits
//

MATH_FUNC inline int numeric_limits<int>::min()
{
    return INT_MIN;
}

MATH_FUNC inline int numeric_limits<int>::lowest()
{
    return INT_MIN;
}

MATH_FUNC inline int numeric_limits<int>::max()
{
    return INT_MAX;
}


//-------------------------------------------------------------------------------------------------
// unsigned int limits
//

MATH_FUNC inline unsigned numeric_limits<unsigned>::min()
{
    return 0;
}

MATH_FUNC inline unsigned numeric_limits<unsigned>::lowest()
{
    return 0;
}

MATH_FUNC inline unsigned numeric_limits<unsigned>::max()
{
    return UINT_MAX;
}


//-------------------------------------------------------------------------------------------------
// float limits
//

MATH_FUNC inline float numeric_limits<float>::min()
{
    return FLT_MIN;
}

MATH_FUNC inline float numeric_limits<float>::lowest()
{
    return -FLT_MAX;
}

MATH_FUNC inline float numeric_limits<float>::max()
{
    return FLT_MAX;
}

MATH_FUNC inline float numeric_limits<float>::epsilon()
{
    return FLT_EPSILON;
}


//-------------------------------------------------------------------------------------------------
// double limits
//

MATH_FUNC inline double numeric_limits<double>::min()
{
    return DBL_MIN;
}

MATH_FUNC inline double numeric_limits<double>::lowest()
{
    return -DBL_MAX;
}

MATH_FUNC inline double numeric_limits<double>::max()
{
    return DBL_MAX;
}

MATH_FUNC inline double numeric_limits<double>::epsilon()
{
    return DBL_EPSILON;
}


//-------------------------------------------------------------------------------------------------
// unorm limits
//

template <unsigned Bits>
MATH_FUNC inline unorm<Bits> numeric_limits<unorm<Bits>>::min()
{
    return unorm<Bits>(0.0f);
}

template <unsigned Bits>
MATH_FUNC inline unorm<Bits> numeric_limits<unorm<Bits>>::lowest()
{
    return unorm<Bits>(0.0f);
}

template <unsigned Bits>
MATH_FUNC inline unorm<Bits> numeric_limits<unorm<Bits>>::max()
{
    return unorm<Bits>(1.0f);
}


//-------------------------------------------------------------------------------------------------
// simd::float4
//

MATH_FUNC inline simd::float4 numeric_limits<simd::float4>::min()
{
    return simd::float4(FLT_MIN);
}

MATH_FUNC inline simd::float4 numeric_limits<simd::float4>::lowest()
{
    return simd::float4(-FLT_MAX);
}

MATH_FUNC inline simd::float4 numeric_limits<simd::float4>::max()
{
    return simd::float4(FLT_MAX);
}

MATH_FUNC inline simd::float4 numeric_limits<simd::float4>::epsilon()
{
    return simd::float4(FLT_EPSILON);
}


//-------------------------------------------------------------------------------------------------
// simd::int4
//

MATH_FUNC inline simd::int4 numeric_limits<simd::int4>::min()
{
    return simd::int4(INT_MIN);
}

MATH_FUNC inline simd::int4 numeric_limits<simd::int4>::lowest()
{
    return simd::int4(INT_MIN);
}

MATH_FUNC inline simd::int4 numeric_limits<simd::int4>::max()
{
    return simd::int4(INT_MAX);
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// simd::float8
//

MATH_FUNC inline simd::float8 numeric_limits<simd::float8>::min()
{
    return simd::float8(FLT_MIN);
}

MATH_FUNC inline simd::float8 numeric_limits<simd::float8>::lowest()
{
    return simd::float8(-FLT_MAX);
}

MATH_FUNC inline simd::float8 numeric_limits<simd::float8>::max()
{
    return simd::float8(FLT_MAX);
}

MATH_FUNC inline simd::float8 numeric_limits<simd::float8>::epsilon()
{
    return simd::float8(FLT_EPSILON);
}


//-------------------------------------------------------------------------------------------------
// simd::int8
//

MATH_FUNC inline simd::int8 numeric_limits<simd::int8>::min()
{
    return simd::int8(INT_MIN);
}

MATH_FUNC inline simd::int8 numeric_limits<simd::int8>::lowest()
{
    return simd::int8(INT_MIN);
}

MATH_FUNC inline simd::int8 numeric_limits<simd::int8>::max()
{
    return simd::int8(INT_MAX);
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

} // MATH_NAMESPACE
