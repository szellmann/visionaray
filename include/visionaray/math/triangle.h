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

    typedef T scalar_type;
    typedef vector<Dim, T> vec_type;

    vec_type v1;
    vec_type e1;
    vec_type e2;
};

} // MATH_NAMESPACE

#endif // VSNRAY_MATH_TRIANGLE_H


