// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_PIXEL_UNPACK_BUFFER_RT
#define VSNRAY_COMMON_PIXEL_UNPACK_BUFFER_RT 1

#include <memory>

#include <GL/glew.h>

#include <visionaray/cuda/graphics_resource.h>
#include <visionaray/math/forward.h>
#include <visionaray/math/vector.h>
#include <visionaray/detail/macros.h>
#include <visionaray/pixel_traits.h>
#include <visionaray/render_target.h>

#include "gl/compositing.h"
#include "gl/handle.h"

namespace visionaray
{

template <
    pixel_format ColorFormat,
    pixel_format DepthFormat,
    pixel_format AccumFormat = PF_UNSPECIFIED
    >
class pixel_unpack_buffer_rt : public render_target
{
public:

    using color_type    = typename pixel_traits<ColorFormat>::type;
    using depth_type    = typename pixel_traits<DepthFormat>::type;
    using accum_type    = typename pixel_traits<AccumFormat>::type;

    using ref_type      = render_target_ref<ColorFormat, DepthFormat, AccumFormat>;

public:

    pixel_unpack_buffer_rt();
   ~pixel_unpack_buffer_rt();

    color_type* color();
    depth_type* depth();
    accum_type* accum();

    color_type const* color() const;
    depth_type const* depth() const;
    accum_type const* accum() const;

    ref_type ref();

    void clear_color_buffer(vec4 const& color = vec4(0.0f));
    void clear_depth_buffer(float depth = 1.0f);
    void clear_accum_buffer(vec4 const& color = vec4(0.0f));
    void begin_frame();
    void end_frame();
    void resize(int w, int h);
    void display_color_buffer() const;

private:

    std::unique_ptr<gl::depth_compositor> compositor;

    cuda::graphics_resource               color_resource;
    cuda::graphics_resource               depth_resource;
    cuda::graphics_resource               accum_resource;

    gl::buffer                            color_buffer;
    gl::buffer                            depth_buffer;
    gl::buffer                            accum_buffer;

};

} // visionaray

#include "pixel_unpack_buffer_rt.inl"

#endif // VSNRAY_COMMON_PIXEL_UNPACK_BUFFER_RT
