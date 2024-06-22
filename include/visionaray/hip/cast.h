// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HIP_CAST_H
#define VSNRAY_HIP_CAST_H 1

#include <hip/hip_runtime.h>

#include "../gpu/cast.h"

namespace visionaray
{
namespace hip
{

template <typename Dest, typename Source>
VSNRAY_FUNC
inline Dest cast(Source const& value)
{
    return ::visionaray::gpu::cast<Dest>(value);
}

} // hip
} // visionaray

#endif // VSNRAY_HIP_CAST_H
