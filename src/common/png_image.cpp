// This file is distributed under the MIT license.
// See the LICENSE file for details.

#if defined(VSNRAY_HAVE_PNG)

#include <cassert>
#include <csetjmp>

#ifndef NDEBUG
#include <iostream>
#include <ostream>
#endif

#include <png.h>

#include "cfile.h"
#include "png_image.h"

namespace visionaray
{

namespace detail
{

struct png_read_context
{
    png_structp png;
    png_infop info;

    png_read_context()
        : png(0)
        , info(0)
    {
    }

   ~png_read_context()
    {
        png_destroy_read_struct(&png, &info, 0);
    }
};

static void png_error_callback(png_structp png_ptr, png_const_charp msg)
{
    fprintf(stderr, "PNG error: \"%s\"\n", msg);

    longjmp(png_jmpbuf(png_ptr), 1);
}


static void png_warning_callback(png_structp /*png_ptr*/, png_const_charp /*msg*/)
{
    // TODO
}

static int png_num_components(int color_type)
{
    switch (color_type)
    {
    case PNG_COLOR_TYPE_GRAY:
        return 1;
    case PNG_COLOR_TYPE_GRAY_ALPHA:
        return 2;
    case PNG_COLOR_TYPE_RGB:
        return 3;
    case PNG_COLOR_TYPE_RGB_ALPHA:
        return 4;
    }

    return -1;
}

} // detail


png_image::png_image(std::string const& filename)
{
    cfile file(filename.c_str(), "r");

    if (!file.good())
    {
        return;
    }


    detail::png_read_context context;

    context.png = png_create_read_struct(
            PNG_LIBPNG_VER_STRING,
            0 /*user-data*/,
            detail::png_error_callback,
            detail::png_warning_callback
            );

    if (context.png == 0)
    {
        return;
    }

    context.info = png_create_info_struct(context.png);

    if (context.info == 0)
    {
        return;
    }


    png_init_io(context.png, file.get());

    png_read_info(context.png, context.info);

    png_uint_32 w = 0;
    png_uint_32 h = 0;
    int bit_depth = 0;
    int color_type = 0;

    png_get_IHDR(context.png, context.info, &w, &h, &bit_depth, &color_type, 0, 0, 0);


    // expand paletted images to RGB
    if (color_type == PNG_COLOR_TYPE_PALETTE)
    {
        png_set_expand(context.png);
    }

    png_read_update_info(context.png, context.info);

    w           = png_get_image_width(context.png, context.info);
    h           = png_get_image_height(context.png, context.info);
    bit_depth   = png_get_bit_depth(context.png, context.info);
    color_type  = png_get_color_type(context.png, context.info);

    assert( detail::png_num_components(color_type) == 3 ); // TODO

    auto pitch  = w * detail::png_num_components(color_type);

    data_.resize(pitch * h);

    for (png_uint_32 y = 0; y < h; ++y)
    {
        png_bytep row = &data_[y * pitch];
        png_read_rows(context.png, &row, &row, 1);
    }

    png_read_end(context.png, context.info);

    width_  = static_cast<size_t>(w);
    height_ = static_cast<size_t>(h);
}

} // visionaray

#endif // VSNRAY_HAVE_PNG
