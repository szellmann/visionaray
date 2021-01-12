// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_FILL_H
#define VSNRAY_CUDA_FILL_H 1

#include <cstddef>

namespace visionaray
{
namespace cuda
{

void fill(void* ptr, size_t len, void* bytes, unsigned count);

} // cuda
} // visionaray

#endif // VSNRAY_CUDA_FILL_H
