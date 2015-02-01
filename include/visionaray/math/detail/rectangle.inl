// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{


//--------------------------------------------------------------------------------------------------
// rectangle : xywh_layout
//

template <typename T>
inline rectangle<xywh_layout, T>::rectangle()
{
}

template <typename T>
inline rectangle<xywh_layout, T>::rectangle(T x, T y, T w, T h)
    : xywh_layout<T>(x, y, w, h)
{
}

template <typename T>
inline rectangle<xywh_layout, T>::rectangle(T const data[4])
    : xywh_layout<T>(data)
{
}

template <typename T>
inline T* rectangle<xywh_layout, T>::data()
{
    return reinterpret_cast<T*>(this);
}

template <typename T>
inline T const* rectangle<xywh_layout, T>::data() const
{
    return reinterpret_cast<T const*>(this);
}

template <typename T>
inline T& rectangle<xywh_layout, T>::operator[](size_t i)
{
    return data()[i];
}

template <typename T>
inline T const& rectangle<xywh_layout, T>::operator[](size_t i) const
{
    return data()[i];
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T>
bool overlapping(rectangle<xywh_layout, T> const& a, rectangle<xywh_layout, T> const& b)
{

    vector<2, T> a1(a.x, a.y);
    vector<2, T> a2(a.x + a.w, a.y + a.h);

    vector<2, T> b1(b.x, b.y);
    vector<2, T> b2(b.x + b.w, b.y + b.h);

    return !(a1[0] > b2[0] || a2[0] < b1[0] || a1[1] > b2[1] || a2[1] < b1[1]);

}


template <typename T>
inline rectangle<xywh_layout, T> combine(rectangle<xywh_layout, T> const& a,
    rectangle<xywh_layout, T> const& b);

template <typename T>
inline rectangle<xywh_layout, T> intersect(rectangle<xywh_layout, T> const& a,
    rectangle<xywh_layout, T> const& b)
{

    if (overlapping(a, b))
    {
        vector<2, T> a1(a.x, a.y);
        vector<2, T> a2(a.x + a.w, a.y + a.h);

        vector<2, T> b1(b.x, b.y);
        vector<2, T> b2(b.x + b.w, b.y + b.h);

        T x = std::max( a1[0], b1[0] );
        T y = std::max( a1[1], b1[1] );

        return rectangle<xywh_layout, T>
        (
            x, y,
            std::min(a2[0], b2[0]) - x,
            std::min(a2[1], b2[1]) - y
        );
    }
    else
    {
        return rectangle<xywh_layout, T>( T(0), T(0), T(0), T(0) );
    }
}


} // MATH_NAMESPACE


