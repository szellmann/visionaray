// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_IO_H
#define VSNRAY_MATH_IO_H 1

#include <cstddef>
#include <istream>
#include <ostream>
#include <sstream>

#include "simd/type_traits.h"
#include "aabb.h"
#include "fixed.h"
#include "matrix.h"
#include "norm.h"
#include "quaternion.h"
#include "rectangle.h"
#include "vector.h"


namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// SIMD type
//

template <
    typename CharT,
    typename Traits,
    typename VecT,
    typename = typename std::enable_if<is_simd_vector<VecT>::value>::type
    >
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, VecT const& v)
{
    using array_t = aligned_array_t<decltype(convert_to_int(v))>;
    using elem_t  = typename element_type<VecT>::type;
    int vec_size  = num_elements<VecT>::value;

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());


    array_t vals = {};
    store(vals, convert_to_int(v));

    s << '(';
    for (int i = 0; i < vec_size; ++i)
    {
        s << static_cast<elem_t>(vals[i]); // e.g. cast mask element to bool
        if (i < vec_size - 1)
        {
            s << ',';
        }
    }
    s << ')';

    return out << s.str();
}

} // simd


//-------------------------------------------------------------------------------------------------
// fixed point
//

template <typename CharT, typename Traits, unsigned I, unsigned F>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, fixed<I, F> f)
{
    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << static_cast<float>(f);

    return out << s.str();
}


//-------------------------------------------------------------------------------------------------
// normalized floats
//

template <typename CharT, typename Traits, unsigned Bits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, snorm<Bits> u)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << static_cast<float>(u);

    return out << s.str();

}

template <typename CharT, typename Traits, unsigned Bits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, unorm<Bits> u)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << static_cast<float>(u);

    return out << s.str();

}


//-------------------------------------------------------------------------------------------------
// quaternions
//

template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, quaternion<T> const& q)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(' << q.w << ',' << q.x << ',' << q.y << ',' << q.z << ')';

    return out << s.str();

}


//-------------------------------------------------------------------------------------------------
// vectors
//

template <size_t Dim, typename T, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits>& in, vector<Dim, T>& v)
{
    CharT ignore = '\0';

    in >> ignore; // '('
    for (size_t d = 0; d < Dim; ++d)
    {
        in >> v[d];
        if (d < Dim - 1)
        {
            in >> ignore; // ','
        }
    }
    in >> ignore; // ')'

    return in;
}

template <size_t Dim, typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, vector<Dim, T> const& v)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(';
    for (size_t d = 0; d < Dim; ++d)
    {
        s << v[d];
        if (d < Dim - 1)
        {
            s << ',';
        }
    }
    s << ')';

    return out << s.str();

}


//-------------------------------------------------------------------------------------------------
// matrices
//

template <typename T, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits>& in, matrix<4, 4, T>& m)
{

    CharT ignore = '\0';

    in >> ignore; // '('
    in >> m.col0;
    in >> ignore; // ','
    in >> m.col1;
    in >> ignore; // ','
    in >> m.col2;
    in >> ignore; // ','
    in >> m.col3;
    in >> ignore; // ')'

    return in;

}

template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, matrix<2, 2, T> const& m)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(' << m.col0 << ',' << m.col1 << ')';

    return out << s.str();

}

template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, matrix<3, 3, T> const& m)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(' << m.col0 << ',' << m.col1 << ',' << m.col2 << ')';

    return out << s.str();

}

template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, matrix<4, 4, T> const& m)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(' << m.col0 << ',' << m.col1 << ',' << m.col2 << ',' << m.col3 << ')';

    return out << s.str();

}

template <size_t N, size_t M, typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, matrix<N, M, T> const& m)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(';
    for (size_t i = 0; i < N; ++i)
    {
        s << m.cols[i];
        if (i < N - 1)
        {
            s << ',';
        }
    }
    s << ')';

    return out << s.str();

}


//-------------------------------------------------------------------------------------------------
// rects
//

template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, rectangle<xywh_layout<T>, T> const& r)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(' << r.x << ',' << r.y << ',' << r.w << ',' << r.h << ')';

    return out << s.str();

}


//-------------------------------------------------------------------------------------------------
// aabb
//

template <typename T, size_t Dim, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, basic_aabb<T, Dim> const& box)
{
    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << box.min << box.max;

    return out << s.str();
}

} // MATH_NAMESPACE

#endif // VSNRAY_MATH_IO_H
