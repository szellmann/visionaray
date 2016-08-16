// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MAKE_UNIQUE_H
#define VSNRAY_COMMON_MAKE_UNIQUE_H 1

#include <memory>
#include <utility>

namespace visionaray
{

template <typename T, typename ...Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // visionaray

#endif // VSNRAY_COMMON_MAKE_UNIQUE_H
