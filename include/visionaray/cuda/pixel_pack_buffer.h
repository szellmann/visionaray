// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_PIXEL_PACK_BUFFER_H
#define VSNRAY_CUDA_PIXEL_PACK_BUFFER_H 1

#include <memory>

#include <visionaray/math/math.h>
#include <visionaray/pixel_format.h>

namespace visionaray
{
namespace cuda
{

class pixel_pack_buffer
{
public:

    pixel_pack_buffer();

    void map(recti viewport, pixel_format format);
    void unmap();

    void const* data() const;

private:

    struct impl;
    std::unique_ptr<impl> const impl_;

};

} // cuda
} // visionaray

#include "detail/pixel_pack_buffer.inl"

#endif // VSNRAY_CUDA_PIXEL_PACK_BUFFER_H
