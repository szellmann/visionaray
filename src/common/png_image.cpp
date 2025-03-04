// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#if VSNRAY_COMMON_HAVE_PNG
#include <png.h>
#endif

#include <visionaray/detail/macros.h>

#include "cfile.h"
#include "png_image.h"

namespace visionaray
{

#if VSNRAY_COMMON_HAVE_PNG
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

struct png_write_context
{
    png_structp png;
    png_infop info;
    png_bytep row;

    png_write_context()
        : png(0)
        , info(0)
        , row(0)
    {
    }

   ~png_write_context()
    {
        png_free_data(png, info, PNG_FREE_ALL, -1);
        png_destroy_write_struct(&png, &info);

        delete[] row;
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
#endif


//-------------------------------------------------------------------------------------------------
// png_image
//

png_image::png_image(int width, int height, pixel_format format, uint8_t const* data)
    : image_base(width, height, format, data)
{
}

bool png_image::load(std::string const& filename)
{
#if VSNRAY_COMMON_HAVE_PNG
    cfile file(filename.c_str(), "r");

    if (!file.good())
    {
        return false;
    }


    png_read_context context;

    context.png = png_create_read_struct(
            PNG_LIBPNG_VER_STRING,
            0 /*user-data*/,
            png_error_callback,
            png_warning_callback
            );

    if (context.png == 0)
    {
        return false;
    }

    context.info = png_create_info_struct(context.png);

    if (context.info == 0)
    {
        return false;
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

    // Only support 8-bit and 16-bit per pixel images
    if (bit_depth != 8 && bit_depth != 16)
    {
        return false;
    }

    auto num_components = png_num_components(color_type);

    switch (num_components)
    {
    case 3:
        format_ = bit_depth == 8 ? PF_RGB8 : PF_RGB16UI;
        break;

    case 4:
        format_ = bit_depth == 8 ? PF_RGBA8 : PF_RGBA16UI;
        break;

    default:
        format_ = PF_UNSPECIFIED;
        return false;
    }

    auto pitch = png_get_rowbytes(context.png, context.info);

    data_.resize(pitch * h);

    for (png_uint_32 y = 0; y < h; ++y)
    {
        png_bytep row = data_.data() + (h - 1) * pitch - y * pitch;
        png_read_rows(context.png, &row, nullptr, 1);
    }

    png_read_end(context.png, context.info);

    width_  = static_cast<int>(w);
    height_ = static_cast<int>(h);

    return true;
#else
    VSNRAY_UNUSED(filename);

    return false;
#endif
}

bool png_image::save(std::string const& filename, file_base::save_options const& /*options*/)
{
#if VSNRAY_COMMON_HAVE_PNG
    cfile file(filename.c_str(), "wb");

    if (!file.good())
    {
        return false;
    }


    png_write_context context;

    context.png = png_create_write_struct(
            PNG_LIBPNG_VER_STRING,
            0 /*user-data*/,
            png_error_callback,
            png_warning_callback
            );

    if (context.png == 0)
    {
        return false;
    }

    context.info = png_create_info_struct(context.png);

    if (context.info == 0)
    {
        return false;
    }


    png_init_io(context.png, file.get());

    // TODO: support other formats than RGB8
    png_set_IHDR(
            context.png,
            context.info,
            width_,
            height_,
            8,
            PNG_COLOR_TYPE_RGB,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_BASE,
            PNG_FILTER_TYPE_BASE
            );

    png_write_info(context.png, context.info);

    size_t pitch = width_ * 3;
    context.row = new png_byte[pitch];

    for (int y = 0; y < height_; ++y)
    {
        for (int x = 0; x < width_; ++x)
        {
            std::memcpy(
                context.row + x * 3,
                data() + (y * width_ + x) * 3,
                3
                );
        }
        png_write_row(context.png, context.row);
    }

    png_write_end(context.png, 0);

    return true;

#else
    VSNRAY_UNUSED(filename);

    return false;
#endif
}

} // visionaray
