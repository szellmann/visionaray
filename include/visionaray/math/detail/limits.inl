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
constexpr T numeric_limits<T>::min()
{
    return std::numeric_limits<T>::min();
}

template <typename T>
constexpr T numeric_limits<T>::lowest()
{
    return std::numeric_limits<T>::lowest();
}

template <typename T>
constexpr T numeric_limits<T>::max()
{
    return std::numeric_limits<T>::max();
}

template <typename T>
constexpr T numeric_limits<T>::epsilon()
{
    return std::numeric_limits<T>::epsilon();
}


//-------------------------------------------------------------------------------------------------
// int limits
//

MATH_FUNC constexpr int numeric_limits<int>::min()
{
    return INT_MIN;
}

MATH_FUNC constexpr int numeric_limits<int>::lowest()
{
    return INT_MIN;
}

MATH_FUNC constexpr int numeric_limits<int>::max()
{
    return INT_MAX;
}


//-------------------------------------------------------------------------------------------------
// unsigned int limits
//

MATH_FUNC constexpr unsigned numeric_limits<unsigned>::min()
{
    return 0;
}

MATH_FUNC constexpr unsigned numeric_limits<unsigned>::lowest()
{
    return 0;
}

MATH_FUNC constexpr unsigned numeric_limits<unsigned>::max()
{
    return UINT_MAX;
}


//-------------------------------------------------------------------------------------------------
// float limits
//

MATH_FUNC constexpr float numeric_limits<float>::min()
{
    return FLT_MIN;
}

MATH_FUNC constexpr float numeric_limits<float>::lowest()
{
    return -FLT_MAX;
}

MATH_FUNC constexpr float numeric_limits<float>::max()
{
    return FLT_MAX;
}

MATH_FUNC constexpr float numeric_limits<float>::epsilon()
{
    return FLT_EPSILON;
}


//-------------------------------------------------------------------------------------------------
// double limits
//

MATH_FUNC constexpr double numeric_limits<double>::min()
{
    return DBL_MIN;
}

MATH_FUNC constexpr double numeric_limits<double>::lowest()
{
    return -DBL_MAX;
}

MATH_FUNC constexpr double numeric_limits<double>::max()
{
    return DBL_MAX;
}

MATH_FUNC constexpr double numeric_limits<double>::epsilon()
{
    return DBL_EPSILON;
}

} // MATH_NAMESPACE
