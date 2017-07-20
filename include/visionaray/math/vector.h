// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_VECTOR_H
#define VSNRAY_MATH_VECTOR_H 1

#include <cstddef>

#include "forward.h"


namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// vector2
//

template <typename T>
class vector<2, T>
{
public:

    typedef T value_type;

    T x;
    T y;

    MATH_FUNC vector() = default;
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

    MATH_FUNC T& operator[](size_t i);
    MATH_FUNC T const& operator[](size_t i) const;

};


//--------------------------------------------------------------------------------------------------
// vector3
//

template <typename T>
class vector<3, T>
{
public:

    typedef T value_type;

    T x;
    T y;
    T z;

    MATH_FUNC vector() = default;
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

    MATH_FUNC T& operator[](size_t i);
    MATH_FUNC T const& operator[](size_t i) const;

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

    typedef T value_type;

    T x;
    T y;
    T z;
    T w;

    MATH_FUNC vector() = default;
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

    MATH_FUNC T& operator[](size_t i);
    MATH_FUNC T const& operator[](size_t i) const;

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

    using value_type = T;

public:

    MATH_FUNC vector() = default;

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
// vector<3, float> with 16-byte alignment
#include "detail/vector3f.inl"
#include "detail/vector4.inl"
// vector<4, float> with 16-byte alignment
#include "detail/vector4f.inl"

#endif // VSNRAY_MATH_VECTOR_H
