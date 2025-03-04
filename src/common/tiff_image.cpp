// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <cstdint>

#if VSNRAY_COMMON_HAVE_TIFF
#include <tiffio.h>
#endif

#include <visionaray/detail/macros.h>

#include "tiff_image.h"

namespace visionaray
{

#if VSNRAY_COMMON_HAVE_TIFF

//-------------------------------------------------------------------------------------------------
// RAII wrapper for tiff files
//

class tiff_file
{
public:

    tiff_file(std::string const& filename, std::string const& mode)
    {
        tiff_ = TIFFOpen(filename.c_str(), mode.c_str());
    }

   ~tiff_file()
    {
        TIFFClose(tiff_);
    }

    TIFF* get() const { return tiff_; }
    bool good() const { return tiff_ != 0; }

private:

    TIFF* tiff_;

};

#endif

bool tiff_image::load(std::string const& filename)
{
#if VSNRAY_COMMON_HAVE_TIFF
    tiff_file file(filename.c_str(), "r");

    if (!file.good())
    {
        return false;
    }

    uint32_t w = 0;
    uint32_t h = 0;
    TIFFGetField(file.get(), TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(file.get(), TIFFTAG_IMAGELENGTH, &h);

    format_ = PF_RGBA8; // TODO

    auto pitch = w * 4;

    data_.resize(pitch * h);

    if (TIFFReadRGBAImage(file.get(), w, h, reinterpret_cast<uint32_t*>(data_.data()), 0))
    {
        width_ = static_cast<int>(w);
        height_ = static_cast<int>(h);
    }

    return true;
#else
    VSNRAY_UNUSED(filename);

    return false;
#endif
}

} // visionaray
