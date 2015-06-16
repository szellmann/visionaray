// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_MATH_TRIANGLE_H
#define VSNRAY_MATH_TRIANGLE_H

#include "primitive.h"
#include "vector.h"

namespace MATH_NAMESPACE
{

template <size_t Dim, typename T, typename P>
class basic_triangle : public primitive<P>
{
public:

    using scalar_type =  T;
    using vec_type    =  vector<Dim, T>;

public:

    MATH_FUNC basic_triangle() = default;
    MATH_FUNC basic_triangle(
            vector<Dim, T> const& v1,
            vector<Dim, T> const& e1,
            vector<Dim, T> const& e2
            )
        : v1(v1)
        , e1(e1)
        , e2(e2)
    {
    }

    vec_type v1;
    vec_type e1;
    vec_type e2;
};

} // MATH_NAMESPACE

#endif // VSNRAY_MATH_TRIANGLE_H
