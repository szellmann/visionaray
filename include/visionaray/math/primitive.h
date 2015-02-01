// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_PRIMITIVE_H
#define VSNRAY_MATH_PRIMITIVE_H

namespace MATH_NAMESPACE
{

namespace detail
{
static unsigned const SpherePrimitive   = 0x00;
static unsigned const TrianglePrimitive = 0x01;
} // detail

template <typename T /* (unsigned) int type */>
struct primitive
{
    typedef T id_type;

    id_type geom_id;
    id_type prim_id;
};

} // visionaray

#endif // VSNRAY_MATH_PRIMITIVE_H


