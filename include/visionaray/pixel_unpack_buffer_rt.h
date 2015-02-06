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

    typedef pixel_traits<PF_RGBA32F>        color_traits;
    typedef pixel_traits<PF_UNSPECIFIED>    depth_traits;
    typedef typename color_traits::type     color_type;
    typedef typename depth_traits::type     depth_type;

    pixel_unpack_buffer_rt();
   ~pixel_unpack_buffer_rt();

    color_type* color();
    depth_type* depth();

    color_type const* color() const;
    depth_type const* depth() const;

private:

    struct impl;
    std::unique_ptr<impl> impl_;

    void begin_frame_impl();
    void end_frame_impl();
    void resize_impl(size_t w, size_t h);
    void display_color_buffer_impl() const;

};

} // visionaray

#endif // VSNRAY_PIXEL_UNPACK_BUFFER_RT


