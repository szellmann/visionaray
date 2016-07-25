// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_ARRAY_REF_H
#define VSNRAY_ARRAY_REF_H 1

#include <cassert>
#include <algorithm>
#include <type_traits>
#include <vector>

#include "detail/macros.h"

namespace visionaray
{

//--------------------------------------------------------------------------------------------------
// array_ref
//

template <class T>
class array_ref
{
public:

    using value_type    = T;
    using reference     = value_type&;
    using pointer       = value_type*;
    using iterator      = value_type*;

private:

    // The array data - an external buffer
    pointer data_;
    // The length of the array
    size_t len_;

public:

    enum : size_t { npos = static_cast<size_t>(-1) };

public:

    // Construct an empty array_ref.
    VSNRAY_FUNC array_ref()
        : data_(nullptr)
        , len_(0)
    {
    }

    // Construct a array_ref from a pointer and a length.
    VSNRAY_FUNC array_ref(pointer data, size_t len)
        : data_(data)
        , len_(len)
    {
        assert((data_ || len_ == 0) && "constructing from a nullptr and a non-zero length");
    }

    // Construct from two iterators
    VSNRAY_FUNC array_ref(iterator Begin, iterator End)
        : array_ref(Begin, static_cast<size_t>(End - Begin))
    {
        assert((Begin ? Begin <= End : !End) && "invalid iterators");
    }

    // Returns a pointer to the start of the array.
    VSNRAY_FUNC pointer data() const
    {
        return data_;
    }

    // Returns the length of the array.
    VSNRAY_FUNC size_t size() const
    {
        return len_;
    }

    // Returns whether this array is null or empty.
    VSNRAY_FUNC bool empty() const
    {
        return size() == 0;
    }

    // Returns an iterator to the first element of the array.
    VSNRAY_FUNC iterator begin() const
    {
        return data();
    }

    // Returns an iterator to one element past the last element of the array.
    VSNRAY_FUNC iterator end() const
    {
        return data() + size();
    }

    // Array access.
    VSNRAY_FUNC reference operator[](size_t Index) const
    {
        assert(Index < size() && "index out of range");
        return data()[Index];
    }

    // Returns the first element of the array.
    VSNRAY_FUNC reference front() const
    {
        assert(!empty() && "index out of range");
        return data()[0];
    }

    // Returns the last element of the array.
    VSNRAY_FUNC reference back() const
    {
        assert(!empty() && "index out of range");
        return data()[size() - 1];
    }

    // Returns the first N elements of the array.
    VSNRAY_FUNC array_ref front(size_t N) const
    {
        N = std::min(N, size());
        return { data(), N };
    }

    // Removes the first N elements from the array.
    VSNRAY_FUNC array_ref drop_front(size_t N) const
    {
        N = std::min(N, size());
        return { data() + N, size() - N };
    }

    // Returns the last N elements of the array.
    VSNRAY_FUNC array_ref back(size_t N) const
    {
        N = std::min(N, size());
        return { data() + (size() - N), N };
    }

    // Removes the last N elements from the array.
    VSNRAY_FUNC array_ref drop_back(size_t N) const
    {
        N = std::min(N, size());
        return { data(), size() - N };
    }

    // Returns the subarray [First, Last).
    VSNRAY_FUNC array_ref slice(size_t First, size_t Last = npos) const
    {
        return front(Last).drop_front(First);
    }

    // Returns whether this array is equal to another.
    VSNRAY_FUNC bool equals(array_ref rhs) const
    {
        return size() == rhs.size() && std::equal(begin(), end(), rhs.begin());
    }

    // Lexicographically compare this array with another.
    VSNRAY_FUNC bool less(array_ref rhs) const
    {
        return std::lexicographical_compare(begin(), end(), rhs.begin(), rhs.end());
    }

    // Convert to a std::vector
    template <class A = std::allocator<value_type>>
    std::vector<value_type, A> vec() const
    {
        return std::vector<value_type, A>(begin(), end());
    }

    // Explicitly convert to a std::vector
    template <class A = std::allocator<value_type>>
    explicit operator std::vector<value_type, A>() const
    {
        return vec<A>();
    }
};

template <class T>
using const_array_ref = array_ref<typename std::add_const<T>::type>;


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template <class T>
VSNRAY_FUNC
inline bool operator==(array_ref<T> lhs, array_ref<T> rhs)
{
    return lhs.equals(rhs);
}

template <class T>
VSNRAY_FUNC
inline bool operator!=(array_ref<T> lhs, array_ref<T> rhs)
{
    return !(lhs == rhs);
}

template <class T>
VSNRAY_FUNC
inline bool operator<(array_ref<T> lhs, array_ref<T> rhs)
{
    return lhs.less(rhs);
}

template <class T>
VSNRAY_FUNC
inline bool operator<=(array_ref<T> lhs, array_ref<T> rhs)
{
    return !(rhs < lhs);
}

template <class T>
VSNRAY_FUNC
inline bool operator>(array_ref<T> lhs, array_ref<T> rhs)
{
    return rhs < lhs;
}

template <class T>
VSNRAY_FUNC
inline bool operator>=(array_ref<T> lhs, array_ref<T> rhs)
{
    return !(lhs < rhs);
}

} // visionaray

#endif // VSNRAY_ARRAY_REF_H
