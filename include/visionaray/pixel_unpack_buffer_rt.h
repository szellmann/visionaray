// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_PIXEL_UNPACK_BUFFER_RT
#define VSNRAY_PIXEL_UNPACK_BUFFER_RT

#include <memory>

#include "detail/macros.h"
#include "render_target.h"

namespace visionaray
{

class pixel_unpack_buffer_rt : public render_target
{
public:

    pixel_unpack_buffer_rt(pixel_format cf, pixel_format df);
   ~pixel_unpack_buffer_rt();

    void* color();
    void* depth();

    void const* color() const;
    void const* depth() const;

private:

    VSNRAY_NOT_COPYABLE(pixel_unpack_buffer_rt)

    struct impl;
    std::unique_ptr<impl> impl_;

    void begin_frame_impl();
    void end_frame_impl();
    void resize_impl(size_t w, size_t h);
    void display_color_buffer_impl() const;

};

} // visionaray

#endif // VSNRAY_PIXEL_UNPACK_BUFFER_RT


