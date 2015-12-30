// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_OUTPUT_H
#define VSNRAY_MATH_OUTPUT_H 1

#include <cstddef>
#include <istream>
#include <ostream>
#include <sstream>

#include "quat.h"


namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// SSE types
//

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, basic_float<__m128> const& v)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    VSNRAY_ALIGN(16) float vals[4];
    store(vals, v);
    s << '(' << vals[0] << ',' << vals[1] << ',' << vals[2] << ',' << vals[3] << ')';

    return out << s.str();

}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, basic_int<__m128i> const& v)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    VSNRAY_ALIGN(16) int vals[4];
    store(vals, v);
    s << '(' << vals[0] << ',' << vals[1] << ',' << vals[2] << ',' << vals[3] << ')';

    return out << s.str();

}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, basic_mask<__m128, __m128i> const& v)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    VSNRAY_ALIGN(16) int vals[4];
    store(vals, v.i);
    s << '(' << static_cast<bool>(vals[0]) << ','
             << static_cast<bool>(vals[1]) << ','
             << static_cast<bool>(vals[2]) << ','
             << static_cast<bool>(vals[3]) << ')';

    return out << s.str();

}


//-------------------------------------------------------------------------------------------------
// AVX types
//

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, basic_float<__m256> const& v)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    VSNRAY_ALIGN(32) float vals[8];
    store(vals, v);
    s << '(' << vals[0] << ',' << vals[1] << ',' << vals[2] << ',' << vals[3] << ','
             << vals[4] << ',' << vals[5] << ',' << vals[6] << ',' << vals[7] << ')';

    return out << s.str();

}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits >& out, basic_int<__m256i> const& v)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    VSNRAY_ALIGN(32) int vals[8];
    store(vals, v);
    s << '(' << vals[0] << ',' << vals[1] << ',' << vals[2] << ',' << vals[3] << ','
             << vals[4] << ',' << vals[5] << ',' << vals[6] << ',' << vals[7] << ')';

    return out << s.str();

}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, basic_mask<__m256, __m256i> const& v)
{

    std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    VSNRAY_ALIGN(32) int vals[8];
    store(vals, v.i);
    s << '(' << static_cast<bool>(vals[0]) << ','
             << static_cast<bool>(vals[1]) << ','
             << static_cast<bool>(vals[2]) << ','
             << static_cast<bool>(vals[3]) << ','
             << static_cast<bool>(vals[4]) << ','
             << static_cast<bool>(vals[5]) << ','
             << static_cast<bool>(vals[6]) << ','
             << static_cast<bool>(vals[7]) << ')';

    return out << s.str();

}

#endif

} // simd


//-------------------------------------------------------------------------------------------------
// quaternions
//

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, quat q)
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

#endif // VSNRAY_MATH_OUTPUT_H
