// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_MATH_SPHERE_H
#define VSNRAY_MATH_SPHERE_H

#include "primitive.h"
#include "vector.h"

namespace MATH_NAMESPACE
{

template <typename T, typename P>
class basic_sphere : public primitive<P>
{
public:

    typedef T scalar_type;
    typedef vector<3, T> vec_type;

    vec_type center;
    scalar_type radius;

};

} // MATH_NAMESPACE

#endif // VSNRAY_MATH_TRIANGLE_H


