// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#if VSNRAY_COMMON_HAVE_TIFF
#include <tiffio.h>
#endif

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

    uint32 w = 0;
    uint32 h = 0;
    TIFFGetField(file.get(), TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(file.get(), TIFFTAG_IMAGELENGTH, &h);

    format_ = PF_RGBA8; // TODO

    auto pitch = w * 4;

    data_.resize(pitch * h);

    if (TIFFReadRGBAImage(file.get(), w, h, reinterpret_cast<uint32*>(data_.data()), 0))
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
