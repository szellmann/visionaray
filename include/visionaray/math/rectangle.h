// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_RECTANGLE_H
#define VSNRAY_MATH_RECTANGLE_H 1

#include <cstddef>

#include "vector.h"


namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// Layout policies
//

template <size_t Dim, typename T>
class min_max_layout
{
public:

    typedef vector<Dim, T> vector_type;

    min_max_layout() = default;
    MATH_FUNC min_max_layout(vector_type const& min, vector_type const& max) : min(min), max(max) {}
    MATH_FUNC min_max_layout(T x, T y, T w, T h) : min(x, y), max(x + w, y + h) {}
    vector_type min;
    vector_type max;
};

template <typename T>
class xywh_layout
{
public:

    xywh_layout() = default;
    MATH_FUNC xywh_layout(T x, T y, T w, T h) : x(x), y(y), w(w), h(h) {}

    MATH_FUNC explicit xywh_layout(T const data[4]) : x(data[0]), y(data[1]), w(data[2]), h(data[3]) {}

    T x;
    T y;
    T w;
    T h;
};


//-------------------------------------------------------------------------------------------------
// rectangle
//

template <typename Layout, typename T>
class rectangle;


//-------------------------------------------------------------------------------------------------
// rectangle with min/max layout
//

template <typename T>
class rectangle<min_max_layout<2, T>, T> : public min_max_layout<2, T>
{
public:

    typedef T value_type;


    rectangle() = default;
    MATH_FUNC rectangle(T x, T y, T w, T h);
    MATH_FUNC rectangle(vector<2, T> const& min, vector<2, T> const& max);

    MATH_FUNC void invalidate();

    MATH_FUNC bool invalid() const;
    MATH_FUNC bool valid() const;

    MATH_FUNC bool empty() const;

    MATH_FUNC bool contains(vector<2, T> const& v) const;
    MATH_FUNC bool contains(rectangle const& r) const;

    MATH_FUNC void insert(vector<2, T> const& v);
    MATH_FUNC void insert(rectangle const& r);

};


//-------------------------------------------------------------------------------------------------
// rectangle with xywh layout
//

template <typename T>
class rectangle<xywh_layout<T>, T> : public xywh_layout<T>
{
public:

    typedef T value_type;


    rectangle() = default;
    MATH_FUNC rectangle(T x, T y, T w, T h);

    MATH_FUNC explicit rectangle(T const data[4]);

    MATH_FUNC T* data();
    MATH_FUNC T const* data() const;

    MATH_FUNC T& operator[](size_t i);
    MATH_FUNC T const& operator[](size_t i) const;

    MATH_FUNC void invalidate();

    MATH_FUNC bool invalid() const;
    MATH_FUNC bool valid() const;

    MATH_FUNC bool empty() const;

    MATH_FUNC bool contains(vector<2, T> const& v) const;
    MATH_FUNC bool contains(rectangle const& r) const;

};

} // MATH_NAMESPACE

#include "detail/rectangle.inl"

#endif // VSNRAY_MATH_RECTANGLE
