// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_VECTOR_H
#define VSNRAY_MATH_VECTOR_H 1

#include <cstddef>

#include "forward.h"

namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// vector traits
//

template <size_t Dim, typename T>
struct vector_traits
{
    using value_type                = T;
    using reference                 = T&;
    using const_reference           = T const&;
    using pointer                   = T*;
    using const_pointer             = T const*;

    using type                      = vector<Dim, T>;
    using reference_to_vector       = vector<Dim, T>&;
    using const_reference_to_vector = vector<Dim, T> const&;
    using pointer_to_vector         = vector<Dim, T>*;
    using const_pointer_to_vector   = vector<Dim, T> const*;
};


//--------------------------------------------------------------------------------------------------
// vector2
//

template <typename T>
class vector<2, T>
{
public:
    using value_type      = typename vector_traits<2, T>::value_type;
    using reference       = typename vector_traits<2, T>::reference;
    using const_reference = typename vector_traits<2, T>::const_reference;
    using pointer         = typename vector_traits<2, T>::pointer;
    using const_pointer   = typename vector_traits<2, T>::const_pointer;

public:

    T x;
    T y;

    vector() = default;
    MATH_FUNC vector(T const& x, T const& y);

    MATH_FUNC explicit vector(T const& s);
    MATH_FUNC explicit vector(T const data[2]);

    template <typename U>
    MATH_FUNC explicit vector(vector<2, U> const& rhs);

    template <typename U>
    MATH_FUNC explicit vector(vector<3, U> const& rhs);

    template <typename U>
    MATH_FUNC explicit vector(vector<4, U> const& rhs);

    template <typename U>
    MATH_FUNC vector& operator=(vector<2, U> const& rhs);

    MATH_FUNC T* data();
    MATH_FUNC T const* data() const;

    MATH_FUNC T& operator[](unsigned i);
    MATH_FUNC T const& operator[](unsigned i) const;

};


//--------------------------------------------------------------------------------------------------
// vector3
//

template <typename T>
class vector<3, T>
{
public:
    using value_type      = typename vector_traits<3, T>::value_type;
    using reference       = typename vector_traits<3, T>::reference;
    using const_reference = typename vector_traits<3, T>::const_reference;
    using pointer         = typename vector_traits<3, T>::pointer;
    using const_pointer   = typename vector_traits<3, T>::const_pointer;

public:

    T x;
    T y;
    T z;

    vector() = default;
    MATH_FUNC vector(T const& x, T const& y, T const& z);

    MATH_FUNC explicit vector(T const& s);
    MATH_FUNC explicit vector(T const data[3]);

    template <typename U>
    MATH_FUNC explicit vector(vector<2, U> const& rhs, U const& z);

    template <typename U>
    MATH_FUNC explicit vector(vector<3, U> const& rhs);

    template <typename U>
    MATH_FUNC explicit vector(vector<4, U> const& rhs);

    template <typename U>
    MATH_FUNC vector& operator=(vector<3, U> const& rhs);

    MATH_FUNC T* data();
    MATH_FUNC T const* data() const;

    MATH_FUNC T& operator[](unsigned i);
    MATH_FUNC T const& operator[](unsigned i) const;

    MATH_FUNC vector<2, T>& xy();
    MATH_FUNC vector<2, T> const& xy() const;

};


//--------------------------------------------------------------------------------------------------
// vector4
//

template <typename T>
class vector<4, T>
{
public:
    using value_type      = typename vector_traits<4, T>::value_type;
    using reference       = typename vector_traits<4, T>::reference;
    using const_reference = typename vector_traits<4, T>::const_reference;
    using pointer         = typename vector_traits<4, T>::pointer;
    using const_pointer   = typename vector_traits<4, T>::const_pointer;

public:

    T x;
    T y;
    T z;
    T w;

    vector() = default;
    MATH_FUNC vector(T const& x, T const& y, T const& z, T const& w);

    MATH_FUNC explicit vector(T const& s);
    MATH_FUNC explicit vector(T const data[4]);

    template <typename U>
    MATH_FUNC explicit vector(vector<2, U> const& rhs, U const& z, U const& w);

    template <typename U>
    MATH_FUNC explicit vector(vector<2, U> const& first, vector<2, U> const& second);

    template <typename U>
    MATH_FUNC explicit vector(vector<3, U> const& rhs, U const& w);

    template <typename U>
    MATH_FUNC explicit vector(vector<4, U> const& rhs);

    template <typename U>
    MATH_FUNC vector& operator=(vector<4, U> const& rhs);

    MATH_FUNC T* data();
    MATH_FUNC T const* data() const;

    MATH_FUNC T& operator[](unsigned i);
    MATH_FUNC T const& operator[](unsigned i) const;

    MATH_FUNC vector<2, T>& xy();
    MATH_FUNC vector<2, T> const& xy() const;

    MATH_FUNC vector<3, T>& xyz();
    MATH_FUNC vector<3, T> const& xyz() const;

};


//-------------------------------------------------------------------------------------------------
// vectorN
//

template <size_t Dim, typename T>
class vector
{
public:
    using value_type      = typename vector_traits<Dim, T>::value_type;
    using reference       = typename vector_traits<Dim, T>::reference;
    using const_reference = typename vector_traits<Dim, T>::const_reference;
    using pointer         = typename vector_traits<Dim, T>::pointer;
    using const_pointer   = typename vector_traits<Dim, T>::const_pointer;

public:

    vector() = default;

    MATH_FUNC explicit vector(T const& s);
    MATH_FUNC explicit vector(T const* data/*[Dim]*/);

    template <typename U>
    MATH_FUNC explicit vector(vector<Dim, U> const& rhs);

    // Dim1 + Dim2 = Dim!
    template <size_t Dim1, size_t Dim2, typename U>
    MATH_FUNC explicit vector(vector<Dim1, U> const& first, vector<Dim2, U> const& second);

    template <typename U>
    MATH_FUNC vector& operator=(vector<Dim, U> const& rhs);

    MATH_FUNC T* data();
    MATH_FUNC T const* data() const;

    MATH_FUNC T& operator[](size_t i);
    MATH_FUNC T const& operator[](size_t i) const;

private:

    T data_[Dim];

};

} // MATH_NAMESPACE

#include "detail/vector.inl"
#include "detail/vector2.inl"
#include "detail/vector3.inl"
#include "detail/vector4.inl"
// vector<4, float> with 16-byte alignment
#include "detail/vector4f.inl"

#endif // VSNRAY_MATH_VECTOR_H
