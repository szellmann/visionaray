// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VIEWER_HOST_DEVICE_RT_H
#define VSNRAY_VIEWER_HOST_DEVICE_RT_H 1

#include <memory>

#include <visionaray/render_target.h>

#include "render_state.h"

namespace visionaray
{

class host_device_rt : public render_target
{
public:

    using color_type = typename pixel_traits<PF_RGBA32F>::type;
    using ref_type = render_target_ref<PF_RGBA32F, PF_UNSPECIFIED>;

public:

    host_device_rt(render_state const& state);
   ~host_device_rt();

    color_type const* color() const;

    ref_type ref();

    void clear_color_buffer(vec4 const& color = vec4(0.0f));
    void begin_frame();
    void end_frame();
    void resize(int w, int h);
    void display_color_buffer() const;

private:

    struct impl;
    std::unique_ptr<impl> impl_;

};

} // visionaray

#endif // VSNRAY_VIEWER_HOST_DEVICE_RT_H
