// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_ALIGNED_VECTOR_H
#define VSNRAY_ALIGNED_VECTOR_H 1

#include <cstddef>
#include <vector>

#include "detail/aligned_allocator.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// An std::vector that is aligned to A byte boundaries
//

template <typename T, size_t A = 16>
using aligned_vector = std::vector<T, aligned_allocator<T, A>>;

} // visionaray

#endif // VSNRAY_ALIGNED_VECTOR_H
