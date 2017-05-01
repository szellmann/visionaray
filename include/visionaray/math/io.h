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

namespace detail
{

// helper function to convert SIMD mask to SIMD int

template <typename T>
T int_from_mask(T const& a)
{
    return a;
}

template <typename F, typename I>
basic_int<I> int_from_mask(basic_mask<F, I> const& m)
{
    return m.i;
}

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
inline basic_int<__m512i> int_from_mask(basic_mask<__mmask16> const& m)
{
    VSNRAY_ALIGN(64) int arr[16];
    store(arr, m);
    return basic_int<__m512i>(arr);
}
#endif

} // detail

template <
    typename CharT,
    typename Traits,
    typename VecT,
    typename = typename std::enable_if<is_simd_vector<VecT>::value>::type
    >
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, VecT const& v)
{
    using array_t = aligned_array_t<decltype(detail::int_from_mask(v))>;
    using elem_t  = typename element_type<VecT>::type;
    int vec_size  = num_elements<VecT>::value;

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());


    array_t vals = {};
    store(vals, detail::int_from_mask(v));

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
operator<<(std::basic_ostream<CharT, Traits>& out, vector<Dim, T> v)
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


template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, vector<3, T> v)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(' << v.x << ',' << v.y << ',' << v.z << ')';

    return out << s.str();

}


template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, vector<4, T> v)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(' << v.x << ',' << v.y << ',' << v.z << ',' << v.w << ')';

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



//-------------------------------------------------------------------------------------------------
// rects
//

template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, rectangle<xywh_layout, T> const& r)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(' << r.x << ',' << r.y << ',' << r.w << ',' << r.h << ')';

    return out << s.str();

}

} // MATH_NAMESPACE

#endif // VSNRAY_MATH_IO_H
