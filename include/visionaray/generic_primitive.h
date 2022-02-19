// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GENERIC_PRIMITIVE_H
#define VSNRAY_GENERIC_PRIMITIVE_H 1

#include "detail/macros.h"
#include "variant.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Generic primitive
//
// Type that is based on variant; by using a tagged union, primitive types stored in an
// accelerator like this don't require virtual inheritance. This is a performance optimization
// to avoid virtual inheritance in inner loops
//

template <typename ...Ts>
class generic_primitive;

template <typename T, typename ...Ts>
class generic_primitive<T, Ts...> : public variant<T, Ts...>
{
public:

    using base_type = variant<T, Ts...>;

public:

    generic_primitive() = default;

    template <typename P>
    /* implicit */ generic_primitive(P const& primitive)
        : base_type(primitive)
    {
    }
};

} // visionaray

#include "detail/generic_primitive/get_color.inl"
#include "detail/generic_primitive/get_normal.inl"
#include "detail/generic_primitive/get_tex_coord.inl"
#include "detail/generic_primitive.inl"

#endif // VSNRAY_GENERIC_PRIMITIVE_H
