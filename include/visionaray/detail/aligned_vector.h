// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_ALIGNED_VECTOR_H
#define VSNRAY_DETAIL_ALIGNED_VECTOR_H

#include <vector>

#include "allocator.h"

namespace visionaray
{

template <typename T, size_t A = 16>
using aligned_vector = std::vector<T, aligned_allocator<T, A>>;

} // visionaray

#endif // VSNRAY_ALIGNED_VECTOR_H


