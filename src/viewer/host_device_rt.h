// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VIEWER_HOST_DEVICE_RT_H
#define VSNRAY_VIEWER_HOST_DEVICE_RT_H 1

#include <memory>

#include <visionaray/render_target.h>

namespace visionaray
{

class host_device_rt : public render_target
{
public:

    using color_type = typename pixel_traits<PF_RGBA8>::type;
    using ref_type = render_target_ref<PF_RGBA8, PF_UNSPECIFIED, PF_RGBA32F>;

    enum buffer
    {
        Front,
        Back
    };

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
            bool double_buffering = true,
            bool direct_rendering = true,
            color_space_type color_space = RGB
            );
   ~host_device_rt();

    mode_type& mode();
    mode_type const& mode() const;

    void set_double_buffering(bool double_buffering);
    bool get_double_buffering() const;

    bool& direct_rendering();
    bool const& direct_rendering() const;

    color_space_type& color_space();
    color_space_type const& color_space() const;

    void swap_buffers();

    color_type const* color(buffer buf = Back) const;

    ref_type ref(buffer buf = Back);

    void clear(vec4 const& color = vec4(0.0f), buffer buf = Back);
    void begin_frame(buffer buf = Back);
    void end_frame(buffer buf = Back);
    void resize(int w, int h);
    void display_color_buffer(buffer buf = Front) const;

private:

    struct impl;
    std::unique_ptr<impl> impl_;

};

} // visionaray

#endif // VSNRAY_VIEWER_HOST_DEVICE_RT_H
