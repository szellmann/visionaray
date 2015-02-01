// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_MATH_AXIS_H
#define VSNRAY_MATH_AXIS_H

#include <cstddef>


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


} // MATH_NAMESPACE


#endif // VSNRAY_MATH_AXIS_H


