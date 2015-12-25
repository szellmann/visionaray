// This file is distributed under the MIT license.
// See the LICENSE file for details.

#if defined(VSNRAY_HAVE_TIFF)

#if 1//ndef NDEBUG
#include <iostream>
#include <ostream>
#endif

#include <tiffio.h>

#include "tiff_image.h"

namespace visionaray
{

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

tiff_image::tiff_image(std::string const& filename)
{
    tiff_file file(filename.c_str(), "r");

    if (!file.good())
    {
        return;
    }

    uint32 w = 0;
    uint32 h = 0;
    TIFFGetField(file.get(), TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(file.get(), TIFFTAG_IMAGELENGTH, &h);

    auto pitch = w * 4;

    data_.resize(pitch * h);

    if (TIFFReadRGBAImage(file.get(), w, h, reinterpret_cast<uint32*>(data_.data()), 0))
    {
        width_ = w;
        height_ = h;
    }
}

} // visionaray

#endif // VSNRAY_HAVE_TIFF
