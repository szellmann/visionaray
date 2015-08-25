// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_MATH_AXIS_H
#define VSNRAY_MATH_AXIS_H 1

#include <cstddef>

#include <vector>

namespace MATH_NAMESPACE
{

template <size_t Dim>
class cartesian_axis;


template <>
class cartesian_axis<2>
{
public:

    enum label { X, Y };

    /* implicit */ cartesian_axis(label l) : val_(l) {}

    operator label() const { return val_; }

private:

    label val_;

};


template <>
class cartesian_axis<3>
{
public:

    enum label { X, Y, Z };

    /* implicit */ cartesian_axis(label l) : val_(l) {}

    operator label() const { return val_; }

private:

    label val_;

};


template <size_t Dim>
MATH_FUNC
vector<Dim, float> to_vector(cartesian_axis<Dim> const& a)
{
    vector<Dim, float> result(0.0f);
    result[(typename cartesian_axis<Dim>::label)(a)] = 1.0f;
    return result;
}

} // MATH_NAMESPACE


#endif // VSNRAY_MATH_AXIS_H
