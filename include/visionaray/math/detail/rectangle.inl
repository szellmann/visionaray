// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../limits.h"
#include "../vector.h"

namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// rectangle : min_max_layout
//

template <typename T>
MATH_FUNC
inline rectangle<min_max_layout<2, T>, T>::rectangle(T x, T y, T w, T h)
    : min_max_layout<2, T>(x, y, w, h)
{
}

template <typename T>
MATH_FUNC
inline rectangle<min_max_layout<2, T>, T>::rectangle(vector<2, T> const& min, vector<2, T> const& max)
    : min_max_layout<2, T>(min, max)
{
}

template <typename T>
MATH_FUNC
inline void rectangle<min_max_layout<2, T>, T>::invalidate()
{
    this->min = vector<2, T>(numeric_limits<T>::max());
    this->max = vector<2, T>(numeric_limits<T>::lowest());
}

template <typename T>
MATH_FUNC
inline bool rectangle<min_max_layout<2, T>, T>::invalid() const
{
    return this->min.x > this->max.x || this->min.y > this->max.y;
}

template <typename T>
MATH_FUNC
inline bool rectangle<min_max_layout<2, T>, T>::valid() const
{
    return this->min.x <= this->max.x && this->min.y <= this->max.y;
}

template <typename T>
MATH_FUNC
inline bool rectangle<min_max_layout<2, T>, T>::empty() const
{
    return this->min.x >= this->max.x || this->min.y >= this->max.y;
}

template <typename T>
MATH_FUNC
inline bool rectangle<min_max_layout<2, T>, T>::contains(vector<2, T> const& v) const
{
    return v.x >= this->min.x && v.x <= this->max.x
        && v.y >= this->min.y && v.y <= this->max.y;
}

template <typename T>
MATH_FUNC
inline bool rectangle<min_max_layout<2, T>, T>::contains(rectangle<min_max_layout<2, T>, T> const& r) const
{
    return contains(r.min) && contains(r.max);
}

template <typename T>
MATH_FUNC
inline void rectangle<min_max_layout<2, T>, T>::insert(vector<2, T> const& v)
{
    this->min = MATH_NAMESPACE::min(this->min, v);
    this->max = MATH_NAMESPACE::max(this->max, v);
}

template <typename T>
MATH_FUNC
inline void rectangle<min_max_layout<2, T>, T>::insert(rectangle<min_max_layout<2, T>, T> const& r)
{
    this->min = MATH_NAMESPACE::min(this->min, r.min);
    this->max = MATH_NAMESPACE::max(this->max, r.max);
}


//--------------------------------------------------------------------------------------------------
// rectangle : xywh_layout
//

template <typename T>
MATH_FUNC
inline rectangle<xywh_layout<T>, T>::rectangle(T x, T y, T w, T h)
    : xywh_layout<T>(x, y, w, h)
{
}

template <typename T>
MATH_FUNC
inline rectangle<xywh_layout<T>, T>::rectangle(T const data[4])
    : xywh_layout<T>(data)
{
}

template <typename T>
MATH_FUNC
inline T* rectangle<xywh_layout<T>, T>::data()
{
    return reinterpret_cast<T*>(this);
}

template <typename T>
MATH_FUNC
inline T const* rectangle<xywh_layout<T>, T>::data() const
{
    return reinterpret_cast<T const*>(this);
}

template <typename T>
MATH_FUNC
inline T& rectangle<xywh_layout<T>, T>::operator[](int i)
{
    return data()[i];
}

template <typename T>
MATH_FUNC
inline T const& rectangle<xywh_layout<T>, T>::operator[](int i) const
{
    return data()[i];
}

template <typename T>
MATH_FUNC
inline void rectangle<xywh_layout<T>, T>::invalidate()
{
    this->x = numeric_limits<T>::max();
    this->y = numeric_limits<T>::max();
    this->w = numeric_limits<T>::lowest();
    this->h = numeric_limits<T>::lowest();
}

template <typename T>
MATH_FUNC
inline bool rectangle<xywh_layout<T>, T>::invalid() const
{
    return this->w < 0 || this->h < 0;
}

template <typename T>
MATH_FUNC
inline bool rectangle<xywh_layout<T>, T>::valid() const
{
    return this->w >= 0 && this->h >= 0;
}

template <typename T>
MATH_FUNC
inline bool rectangle<xywh_layout<T>, T>::empty() const
{
    return this->w <= 0 || this->h <= 0;
}

template <typename T>
MATH_FUNC
inline bool rectangle<xywh_layout<T>, T>::contains(vector<2, T> const& v) const
{
    vector<2, T> min(this->x, this->y);
    vector<2, T> max(this->x + this->w, this->y + this->h);
    return v.x >= min.x && v.x <= max.x && v.y >= min.y && v.y <= max.y;
}

template <typename T>
MATH_FUNC
inline bool rectangle<xywh_layout<T>, T>::contains(rectangle const& r) const
{
    vector<2, T> v1(r.x, r.y);
    vector<2, T> v2(r.x + r.w, r.y + r.h);
    return this->contains(v1) && this->contains(v2);
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

template <typename T>
MATH_FUNC
bool operator==(rectangle<xywh_layout<T>, T> const& a, rectangle<xywh_layout<T>, T> const& b)
{
    return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h;
}

template <typename T>
MATH_FUNC
bool operator!=(rectangle<xywh_layout<T>, T> const& a, rectangle<xywh_layout<T>, T> const& b)
{
    return !(a == b);
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T>
MATH_FUNC
bool overlapping(rectangle<xywh_layout<T>, T> const& a, rectangle<xywh_layout<T>, T> const& b)
{

    vector<2, T> a1(a.x, a.y);
    vector<2, T> a2(a.x + a.w, a.y + a.h);

    vector<2, T> b1(b.x, b.y);
    vector<2, T> b2(b.x + b.w, b.y + b.h);

    return !(a1[0] > b2[0] || a2[0] < b1[0] || a1[1] > b2[1] || a2[1] < b1[1]);

}


template <typename T>
MATH_FUNC
inline rectangle<xywh_layout<T>, T> combine(
        rectangle<xywh_layout<T>, T> const& a,
        rectangle<xywh_layout<T>, T> const& b
        )
{
    vector<2, T> a1(a.x, a.y);
    vector<2, T> a2(a.x + a.w, a.y + a.h);

    vector<2, T> b1(b.x, b.y);
    vector<2, T> b2(b.x + b.w, b.y + b.h);

    vector<2, T> c1(min(a1, b1));
    vector<2, T> c2(max(a2, b2));

    return rectangle<xywh_layout<T>, T>(
            c1.x,
            c1.y,
            c2.x - c1.x,
            c2.y - c1.y
            );
}

template <typename T>
MATH_FUNC
inline rectangle<xywh_layout<T>, T> intersect(
        rectangle<xywh_layout<T>, T> const& a,
        rectangle<xywh_layout<T>, T> const& b
        )
{

    if (overlapping(a, b))
    {
        vector<2, T> a1(a.x, a.y);
        vector<2, T> a2(a.x + a.w, a.y + a.h);

        vector<2, T> b1(b.x, b.y);
        vector<2, T> b2(b.x + b.w, b.y + b.h);

        T x = max( a1[0], b1[0] );
        T y = max( a1[1], b1[1] );

        return rectangle<xywh_layout<T>, T>(
                x, y,
                min(a2[0], b2[0]) - x,
                min(a2[1], b2[1]) - y
                );
    }
    else
    {
        return rectangle<xywh_layout<T>, T>( T(0), T(0), T(0), T(0) );
    }
}

} // MATH_NAMESPACE
