// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_HOST_DEVICE_RT_H
#define VSNRAY_COMMON_HOST_DEVICE_RT_H 1

#include <memory>

#include <visionaray/render_target.h>

namespace visionaray
{

class host_device_rt : public render_target
{
public:

    using color_type = typename pixel_traits<PF_RGBA32F>::type;
    using ref_type = render_target_ref<PF_RGBA32F, PF_UNSPECIFIED>;

    enum mode_type
    {
        CPU,
        GPU
    };

    enum color_space_type
    {
        RGB,
        SRGB
    };

public:

    host_device_rt(
            mode_type mode,
            bool direct_rendering = true,
            color_space_type color_space = RGB
            );
   ~host_device_rt();

    mode_type& mode();
    mode_type const& mode() const;

    bool& direct_rendering();
    bool const& direct_rendering() const;

    color_space_type& color_space();
    color_space_type const& color_space() const;

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

#endif // VSNRAY_COMMON_HOST_DEVICE_RT_H
