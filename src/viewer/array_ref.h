// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cassert>
#include <algorithm>
#include <type_traits>
#include <vector>

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
    pointer Data;
    // The length of the array
    size_t Length;

public:
    enum : size_t { npos = static_cast<size_t>(-1) };

public:
    // Construct an empty array_ref.
    array_ref()
        : Data(nullptr)
        , Length(0)
    {
    }

    // Construct a array_ref from a pointer and a length.
    array_ref(pointer Data, size_t Length)
        : Data(Data)
        , Length(Length)
    {
        assert((Data || Length == 0) && "constructing from a nullptr and a non-zero length");
    }

    // Construct from two iterators
    array_ref(iterator Begin, iterator End)
        : array_ref(Begin, static_cast<size_t>(End - Begin))
    {
        assert((Begin ? Begin <= End : !End) && "invalid iterators");
    }

    // Returns a pointer to the start of the array.
    pointer data() const {
        return Data;
    }

    // Returns the length of the array.
    size_t size() const {
        return Length;
    }

    // Returns whether this array is null or empty.
    bool empty() const {
        return size() == 0;
    }

    // Returns an iterator to the first element of the array.
    iterator begin() const {
        return data();
    }

    // Returns an iterator to one element past the last element of the array.
    iterator end() const {
        return data() + size();
    }

    // Array access.
    reference operator [](size_t Index) const
    {
        assert(Index < size() && "index out of range");
        return data()[Index];
    }

    // Returns the first element of the array.
    reference front() const
    {
        assert(!empty() && "index out of range");
        return data()[0];
    }

    // Returns the last element of the array.
    reference back() const
    {
        assert(!empty() && "index out of range");
        return data()[size() - 1];
    }

    // Returns the first N elements of the array.
    array_ref front(size_t N) const
    {
        N = std::min(N, size());
        return { data(), N };
    }

    // Removes the first N elements from the array.
    array_ref drop_front(size_t N) const
    {
        N = std::min(N, size());
        return { data() + N, size() - N };
    }

    // Returns the last N elements of the array.
    array_ref back(size_t N) const
    {
        N = std::min(N, size());
        return { data() + (size() - N), N };
    }

    // Removes the last N elements from the array.
    array_ref drop_back(size_t N) const
    {
        N = std::min(N, size());
        return { data(), size() - N };
    }

    // Returns the subarray [First, Last).
    array_ref slice(size_t First, size_t Last = npos) const {
        return front(Last).drop_front(First);
    }

    // Returns whether this array is equal to another.
    bool equals(array_ref RHS) const {
        return size() == RHS.size() && std::equal(begin(), end(), RHS.begin());
    }

    // Lexicographically compare this array with another.
    bool less(array_ref RHS) const {
        return std::lexicographical_compare(begin(), end(), RHS.begin(), RHS.end());
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
inline bool operator ==(array_ref<T> LHS, array_ref<T> RHS) {
    return LHS.equals(RHS);
}

template <class T>
inline bool operator !=(array_ref<T> LHS, array_ref<T> RHS) {
    return !(LHS == RHS);
}

template <class T>
inline bool operator <(array_ref<T> LHS, array_ref<T> RHS) {
    return LHS.less(RHS);
}

template <class T>
inline bool operator <=(array_ref<T> LHS, array_ref<T> RHS) {
    return !(RHS < LHS);
}

template <class T>
inline bool operator >(array_ref<T> LHS, array_ref<T> RHS) {
    return RHS < LHS;
}

template <class T>
inline bool operator >=(array_ref<T> LHS, array_ref<T> RHS) {
    return !(LHS < RHS);
}

} // namespace visionaray
