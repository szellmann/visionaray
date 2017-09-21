// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_MANAGED_VECTOR_H
#define VSNRAY_CUDA_MANAGED_VECTOR_H 1

#include "managed_allocator.h"

namespace visionaray
{
namespace cuda
{

//-------------------------------------------------------------------------------------------------
// An std::vector that stores data in CUDA unified memory
//

template <typename T>
using managed_vector = std::vector<T, managed_allocator<T>>;

} // cuda
} // visionaray

#endif // VSNRAY_CUDA_MANAGED_VECTOR_H
