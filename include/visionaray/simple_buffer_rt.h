// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SIMPLE_BUFFER_RT_H
#define VSNRAY_SIMPLE_BUFFER_RT_H 1

#include "aligned_vector.h"
#include "pixel_traits.h"
#include "render_target.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// The most simple render target, only provides buffers for color and depth.
// Does NOT implement display_color_buffer()
//

template <
    pixel_format ColorFormat,
    pixel_format DepthFormat,
    pixel_format AccumFormat = PF_UNSPECIFIED
    >
class simple_buffer_rt : public render_target
{
public:

    using color_type    = typename pixel_traits<ColorFormat>::type;
    using depth_type    = typename pixel_traits<DepthFormat>::type;
    using accum_type    = typename pixel_traits<AccumFormat>::type;

    using ref_type      = render_target_ref<ColorFormat, DepthFormat, AccumFormat>;

public:

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

private:

    aligned_vector<color_type> color_buffer;
    aligned_vector<depth_type> depth_buffer;
    aligned_vector<accum_type> accum_buffer;

};

} // visionaray

#include "detail/simple_buffer_rt.inl"

#endif // VSNRAY_SIMPLE_BUFFER_RT_H
