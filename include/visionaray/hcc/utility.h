// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HCC_UTILITY_H
#define VSNRAY_HCC_UTILITY_H 1

#include <utility>

#include "../detail/macros.h"

namespace visionaray
{
namespace hcc
{

//-------------------------------------------------------------------------------------------------
// hcc::pair
//

template <typename T1, typename T2>
struct pair
{
    T1 first;
    T2 second;

    VSNRAY_FUNC
    pair() = default;

    VSNRAY_FUNC
    pair(T1 const& x, T2 const& y)
        : first(x)
        , second(y)
    {
    }

    // TODO
    //
};


//-------------------------------------------------------------------------------------------------
// hcc::make_pair
//

template <typename T1, typename T2>
VSNRAY_FUNC
pair<T1, T2> make_pair(T1 x, T2 y)
{
    return hcc::pair<T1, T2>(x, y);
}

template <typename T1, typename T2>
VSNRAY_FUNC
pair<T1, T2> make_pair(T1&& x, T2&& y)
{
    return hcc::pair<T1, T2>{std::move(x), std::move(y)};
}

} // hcc
} // visionaray

#endif // VSNRAY_HCC_UTILITY_H
