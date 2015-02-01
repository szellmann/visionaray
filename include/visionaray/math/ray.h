// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_MATH_RAY_H
#define VSNRAY_MATH_RAY_H

namespace MATH_NAMESPACE
{

template <typename T>
class basic_ray
{
public:

    typedef T scalar_type;
    typedef vector<3, T> vec_type;

    vec_type ori;
    vec_type dir;

    MATH_FUNC inline basic_ray() {}
    MATH_FUNC inline basic_ray(vec_type const& o, vec_type const& d) : ori(o), dir(d) {}

};

} // MATH_NAMESPACE

#endif // VSNRAY_MATH_RAY_H


