// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SIMPLE_BUFFER_RT
#define VSNRAY_SIMPLE_BUFFER_RT 1

#include "aligned_vector.h"
#include "pixel_traits.h"
#include "render_target.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// The most simple render target, only provides buffers for color and depth.
// Does NOT implement display_color_buffer()
//

template <pixel_format ColorFormat, pixel_format DepthFormat>
class simple_buffer_rt : public render_target
{
public:

    using color_type    = typename pixel_traits<ColorFormat>::type;
    using depth_type    = typename pixel_traits<DepthFormat>::type;

    using ref_type      = render_target_ref<ColorFormat, DepthFormat>;

public:

    color_type* color();
    depth_type* depth();

    color_type const* color() const;
    depth_type const* depth() const;

    ref_type ref();

    void begin_frame();
    void end_frame();
    void resize(size_t w, size_t h);

private:

    aligned_vector<color_type> color_buffer;
    aligned_vector<depth_type> depth_buffer;

};

} // visionaray

#include "detail/simple_buffer_rt.inl"

#endif // VSNRAY_SIMPLE_BUFFER_RT
