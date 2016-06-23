// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GENERIC_PRIMITIVE_H
#define VSNRAY_GENERIC_PRIMITIVE_H 1

#include "detail/macros.h"
#include "math/math.h"
#include "prim_traits.h"
#include "variant.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Generic primitive
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

#include "detail/generic_primitive.inl"
#include "detail/generic_primitive_get_color.h"
#include "detail/generic_primitive_get_normal.h"
#include "detail/generic_primitive_get_tex_coord.h"

#endif // VSNRAY_GENERIC_PRIMITIVE_H
