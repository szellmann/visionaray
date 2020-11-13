// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// matrix members
//

template <size_t N, size_t M, typename T>
MATH_FUNC
inline T* matrix<N, M, T>::data()
{
    return reinterpret_cast<T*>(this);
}

template <size_t N, size_t M, typename T>
MATH_FUNC
inline T const* matrix<N, M, T>::data() const
{
    return reinterpret_cast<T const*>(this);
}

template <size_t N, size_t M, typename T>
MATH_FUNC
inline vector<N, T>& matrix<N, M, T>::operator()(size_t col)
{
    return *(reinterpret_cast<vector<N, T>*>(this) + col);
}

template <size_t N, size_t M, typename T>
MATH_FUNC
inline vector<N, T> const& matrix<N, M, T>::operator()(size_t col) const
{
    return *(reinterpret_cast<vector<N, T> const*>(this) + col);
}

template <size_t N, size_t M, typename T>
MATH_FUNC
inline T& matrix<N, M, T>::operator()(size_t row, size_t col)
{
    return (operator()(col))[row];
}

template <size_t N, size_t M, typename T>
MATH_FUNC
inline T const& matrix<N, M, T>::operator()(size_t row, size_t col) const
{
    return (operator()(col))[row];
}

template <size_t N, size_t M, typename T>
MATH_FUNC
inline matrix<N, M, T> matrix<N, M, T>::identity()
{
    static_assert(N == M, "Matrix not symmetrical");

    matrix<N, M, T> result;

    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < M; ++j)
        {
            result(i, j) = i == j ? T(1.0) : T(0.0);
        }
    }

    return result;
}


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template <size_t N, size_t M, typename T>
MATH_FUNC
inline bool operator==(matrix<N, M, T> const& a, matrix<N, M, T> const& b)
{
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < M; ++j)
        {
            if (a(i, j) != b(i, j))
            {
                return false;
            }
        }
    }

    return true;
}

template <size_t N, size_t M, typename T>
MATH_FUNC
inline bool operator!=(matrix<N, M, T> const& a, matrix<N, M, T> const& b)
{
    return !(a == b);
}

} // MATH_NAMESPACE
