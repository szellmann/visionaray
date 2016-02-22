// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Accessors
//

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename simple_buffer_rt<ColorFormat, DepthFormat>::color_type* simple_buffer_rt<ColorFormat, DepthFormat>::color()
{
    return color_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename simple_buffer_rt<ColorFormat, DepthFormat>::depth_type* simple_buffer_rt<ColorFormat, DepthFormat>::depth()
{
    return depth_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename simple_buffer_rt<ColorFormat, DepthFormat>::color_type const* simple_buffer_rt<ColorFormat, DepthFormat>::color() const
{
    return color_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename simple_buffer_rt<ColorFormat, DepthFormat>::depth_type const* simple_buffer_rt<ColorFormat, DepthFormat>::depth() const
{
    return depth_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename simple_buffer_rt<ColorFormat, DepthFormat>::ref_type simple_buffer_rt<ColorFormat, DepthFormat>::ref()
{
    return typename simple_buffer_rt<ColorFormat, DepthFormat>::ref_type( color(), depth() );
}


//-------------------------------------------------------------------------------------------------
// Interface
//

template <pixel_format ColorFormat, pixel_format DepthFormat>
void simple_buffer_rt<ColorFormat, DepthFormat>::begin_frame()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void simple_buffer_rt<ColorFormat, DepthFormat>::end_frame()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void simple_buffer_rt<ColorFormat, DepthFormat>::resize(size_t w, size_t h)
{
    render_target::resize(w, h);


    color_buffer.resize(w * h);

    if (DepthFormat != PF_UNSPECIFIED)
    {
        depth_buffer.resize(w * h);
    }
}

} // visionaray
