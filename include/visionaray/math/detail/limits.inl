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
// snorm limits
//

template <unsigned Bits>
MATH_FUNC inline snorm<Bits> numeric_limits<snorm<Bits>>::min()
{
    return snorm<Bits>(0.0f);
}

template <unsigned Bits>
MATH_FUNC inline snorm<Bits> numeric_limits<snorm<Bits>>::lowest()
{
    return snorm<Bits>(-1.0f);
}

template <unsigned Bits>
MATH_FUNC inline snorm<Bits> numeric_limits<snorm<Bits>>::max()
{
    return snorm<Bits>(1.0f);
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
// simd::basic_float
//

template <typename T>
MATH_CPU_FUNC inline simd::basic_float<T> numeric_limits<simd::basic_float<T>>::min()
{
    return simd::basic_float<T>(FLT_MIN);
}

template <typename T>
MATH_CPU_FUNC inline simd::basic_float<T> numeric_limits<simd::basic_float<T>>::lowest()
{
    return simd::basic_float<T>(-FLT_MAX);
}

template <typename T>
MATH_CPU_FUNC inline simd::basic_float<T> numeric_limits<simd::basic_float<T>>::max()
{
    return simd::basic_float<T>(FLT_MAX);
}

template <typename T>
MATH_CPU_FUNC inline simd::basic_float<T> numeric_limits<simd::basic_float<T>>::epsilon()
{
    return simd::basic_float<T>(FLT_EPSILON);
}


//-------------------------------------------------------------------------------------------------
// simd::basic_int
//

template <typename T>
MATH_CPU_FUNC inline simd::basic_int<T> numeric_limits<simd::basic_int<T>>::min()
{
    return simd::basic_int<T>(INT_MIN);
}

template <typename T>
MATH_CPU_FUNC inline simd::basic_int<T> numeric_limits<simd::basic_int<T>>::lowest()
{
    return simd::basic_int<T>(INT_MIN);
}

template <typename T>
MATH_CPU_FUNC inline simd::basic_int<T> numeric_limits<simd::basic_int<T>>::max()
{
    return simd::basic_int<T>(INT_MAX);
}

} // MATH_NAMESPACE
