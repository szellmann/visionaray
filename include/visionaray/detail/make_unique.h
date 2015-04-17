// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_MAKE_UNIQUE_H
#define VSNRAY_DETAIL_MAKE_UNIQUE_H

#include <memory>

namespace visionaray { namespace detail {

template <typename T, typename ...Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}} // namespace visionaray::detail

#endif
