// This file is distributed under the MIT license.
// See the LICENSE file for details.

#if defined(VSNRAY_HAVE_JPEG)

#include <csetjmp>
#include <cstdio>

#include <jpeglib.h>

#include "jpeg_image.h"

namespace visionaray
{

namespace detail
{

struct error_mngr
{
    jpeg_error_mgr pub;
    jmp_buf jmpbuf;
};


static void error_exit_func(j_common_ptr info)
{
    error_mngr* err = reinterpret_cast<error_mngr*>(info->err);
    // TODO: print debug info
    longjmp(err->jmpbuf, 1);
}


//-------------------------------------------------------------------------------------------------
// RAII wrapper for FILE -- jpeg API requires c-style FILE handles
//

class input_file
{
public:

    input_file(std::string const& filename)
    {
        file_ = fopen(filename.c_str(), "r");
    }

   ~input_file()
    {
        fclose(file_);
    }

    FILE* get() const { return file_; }
    bool good() const { return file_ != 0; }

private:

    FILE* file_;

};

struct decompress_ptr
{
    jpeg_decompress_struct* info;

    decompress_ptr() : info(0) {}

   ~decompress_ptr()
    {
        jpeg_finish_decompress(info);
        jpeg_destroy_decompress(info);
    }
};

}

jpeg_image::jpeg_image(std::string const& filename)
{
    detail::input_file file(filename.c_str());

    if (!file.good())
    {
        return;
    }

    struct jpeg_decompress_struct info;
    detail::decompress_ptr info_ptr;
    detail::error_mngr err;

    info.err = jpeg_std_error(&err.pub);
    err.pub.error_exit = detail::error_exit_func;

    if (setjmp(err.jmpbuf))
    {
        return; // TODO?
    }

    jpeg_create_decompress(&info);

    info_ptr.info = &info;

    jpeg_stdio_src(&info, file.get());

    jpeg_read_header(&info, TRUE);
    width_      = info.image_width;
    height_     = info.image_height;
    auto pitch  = width_ * 3;

    jpeg_start_decompress(&info);

    data_.resize(pitch * height_);

    while (info.output_scanline < info.output_height)
    {
        unsigned char* row = data_.data() + (info.output_height - 1 - info.output_scanline) * pitch;

        jpeg_read_scanlines(&info, &row, 1);
    }
}

}

#endif // VSNRAY_HAVE_JPEG


