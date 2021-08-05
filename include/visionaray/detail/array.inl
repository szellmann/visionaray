// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/config.h>

#include <stdexcept>
#include <utility>

#if VSNRAY_HAVE_THRUST
#include <thrust/swap.h>
#endif

#include "compiler.h"

#ifdef VSNRAY_CXX_HAS_CONSTEXPR
#define VSNRAY_CONSTEXPR_ constexpr
#else
#define VSNRAY_CONSTEXPR_
#endif

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// array members
//

template <typename T, size_t N>
VSNRAY_FUNC
inline T& array<T, N>::at(size_t pos)
{
    if (pos >= N)
    {
#if VSNRAY_CPU_MODE
        throw std::out_of_range("Index out of range");
#endif
    }

    return data_[pos];
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T const& array<T, N>::at(size_t pos) const
{
    if (pos >= N)
    {
#if VSNRAY_CPU_MODE
        throw std::out_of_range("Index out of range");
#endif
    }

    return data_[pos];
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T& array<T, N>::operator[](size_t pos)
{
    return data_[pos];
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T const& array<T, N>::operator[](size_t pos) const
{
    return data_[pos];
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T& array<T, N>::front()
{
    return data_[0];
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T const& array<T, N>::front() const
{
    return data_[0];
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T& array<T, N>::back()
{
    return data_[N - 1];
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T const& array<T, N>::back() const
{
    return data_[N - 1];
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T* array<T, N>::data()
{
    return data_;
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T const* array<T, N>::data() const
{
    return data_;
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T* array<T, N>::begin()
{
    return data_;
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T const* array<T, N>::begin() const
{
    return data_;
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T const* array<T, N>::cbegin() const
{
    return data_;
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T* array<T, N>::end()
{
    return data_ + N;
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T const* array<T, N>::end() const
{
    return data_ + N;
}

template <typename T, size_t N>
VSNRAY_FUNC
inline T const* array<T, N>::cend() const
{
    return data_ + N;
}

template <typename T, size_t N>
VSNRAY_FUNC
inline typename array<T, N>::reverse_iterator array<T, N>::rbegin()
{
    return typename array<T, N>::reverse_iterator(data_ + N);
}

template <typename T, size_t N>
VSNRAY_FUNC
inline typename array<T, N>::const_reverse_iterator array<T, N>::rbegin() const
{
    return typename array<T, N>::const_reverse_iterator(data_ + N);
}

template <typename T, size_t N>
VSNRAY_FUNC
inline typename array<T, N>::const_reverse_iterator array<T, N>::crbegin() const
{
    return typename array<T, N>::const_reverse_iterator(data_ + N);
}

template <typename T, size_t N>
VSNRAY_FUNC
inline typename array<T, N>::reverse_iterator array<T, N>::rend()
{
    return typename array<T, N>::reverse_iterator(data_);
}

template <typename T, size_t N>
VSNRAY_FUNC
inline typename array<T, N>::const_reverse_iterator array<T, N>::rend() const
{
    return typename array<T, N>::const_reverse_iterator(data_);
}

template <typename T, size_t N>
VSNRAY_FUNC
inline typename array<T, N>::const_reverse_iterator array<T, N>::crend() const
{
    return typename array<T, N>::const_reverse_iterator(data_);
}


template <typename T, size_t N>
VSNRAY_FUNC
inline VSNRAY_CONSTEXPR_ bool array<T, N>::empty() const
{
    return N == 0;
}

template <typename T, size_t N>
VSNRAY_FUNC
inline VSNRAY_CONSTEXPR_ size_t array<T, N>::size() const
{
    return N;
}

template <typename T, size_t N>
VSNRAY_FUNC
inline VSNRAY_CONSTEXPR_ size_t array<T, N>::max_size() const
{
    return N;
}

template <typename T, size_t N>
VSNRAY_FUNC
inline void array<T, N>::fill(T const& value)
{
    // May not call std::fill() and the like with CUDA
    for (size_t i = 0; i < N; ++i)
    {
        data_[i] = value;
    }
}

template <typename T, size_t N>
VSNRAY_FUNC
inline void array<T, N>::swap(array<T, N>& rhs)
{
#if VSNRAY_HAVE_THRUST
    using thrust::swap;
#else
    using std::swap;
#endif

    for (size_t i = 0; i < N; ++i)
    {
        swap(data_[i], rhs.data_[i]);
    }
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

template <typename T, size_t N>
VSNRAY_FUNC
bool operator==(array<T, N> const& a, array<T, N> const& b)
{
    for (size_t i = 0; i < N; ++i)
    {
        if (a[i] != b[i])
        {
            return false;
        }
    }

    return true;
}

template <typename T, size_t N>
VSNRAY_FUNC
bool operator!=(array<T, N> const& a, array<T, N> const& b)
{
    return !(a == b);
}

// TODO: lexicographic comparisons!

} // visionaray


namespace std
{

//-------------------------------------------------------------------------------------------------
// Element access
//

template <size_t I, typename T, size_t N>
VSNRAY_CONSTEXPR_ T& get(visionaray::array<T, N>& a)
{
    return a[I];
}

template <size_t I, typename T, size_t N>
VSNRAY_CONSTEXPR_ T&& get(visionaray::array<T, N>&& a)
{
    return a[I];
}

template <size_t I, typename T, size_t N>
VSNRAY_CONSTEXPR_ T const& get(visionaray::array<T, N> const& a)
{
    return a[I];
}

template <size_t I, typename T, size_t N>
VSNRAY_CONSTEXPR_ T const&& get(visionaray::array<T, N> const&& a)
{
    return a[I];
}

} // std


#if VSNRAY_HAVE_THRUST

namespace thrust
{

//-------------------------------------------------------------------------------------------------
// Element access
//

template <size_t I, typename T, size_t N>
VSNRAY_FUNC
VSNRAY_CONSTEXPR_ T& get(visionaray::array<T, N>& a)
{
    return a[I];
}

template <size_t I, typename T, size_t N>
VSNRAY_FUNC
VSNRAY_CONSTEXPR_ T&& get(visionaray::array<T, N>&& a)
{
    return a[I];
}

template <size_t I, typename T, size_t N>
VSNRAY_FUNC
VSNRAY_CONSTEXPR_ T const& get(visionaray::array<T, N> const& a)
{
    return a[I];
}

template <size_t I, typename T, size_t N>
VSNRAY_FUNC
VSNRAY_CONSTEXPR_ T const&& get(visionaray::array<T, N> const&& a)
{
    return a[I];
}

} // thrust

#endif

#undef VSNRAY_CONSTEXPR_
