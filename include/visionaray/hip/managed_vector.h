// This file is distributed under the MIT license.
// See the LICENSE file for details.
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.

#pragma once

#ifndef VSNRAY_HIP_MANAGED_VECTOR_H
#define VSNRAY_HIP_MANAGED_VECTOR_H 1

#include "managed_allocator.h"

namespace visionaray
{
namespace hip
{

//-------------------------------------------------------------------------------------------------
// An std::vector that stores data in HIP unified memory
//

template <typename T>
using managed_vector = std::vector<T, managed_allocator<T>>;

} // hip
} // visionaray

#endif // VSNRAY_HIP_MANAGED_VECTOR_H
