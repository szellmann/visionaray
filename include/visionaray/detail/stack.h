// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_STACK_H
#define VSNRAY_DETAIL_STACK_H

#include "macros.h"
   
namespace visionaray
{
namespace detail
{

template <size_t N>
struct stack
{
    VSNRAY_FUNC stack()
        : ptr(0)
    {
    }

    VSNRAY_FUNC bool empty() const
    {
        return ptr == 0;
    }

    VSNRAY_FUNC void push(unsigned v)
    {
        data[++ptr] = v;
    }

    VSNRAY_FUNC unsigned pop()
    {
        return data[ptr--];
    }

    unsigned data[N];
    unsigned ptr;
};

} // detail
} // visionaray


#endif // VSNRAY_DETAIL_STACK_H
