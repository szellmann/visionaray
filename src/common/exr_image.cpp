// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <ostream>

#if VSNRAY_COMMON_HAVE_OPENEXR
#include <ImfArray.h>
#include <ImfFrameBuffer.h>
#include <ImfRgba.h>
#include <ImfRgbaFile.h>
#endif

#include <visionaray/math/forward.h>
#include <visionaray/math/vector.h>
#include <visionaray/aligned_vector.h>

#include "exr_image.h"

namespace visionaray
{

#if VSNRAY_COMMON_HAVE_OPENEXR
static void store_rgb32f(
        aligned_vector<uint8_t>&       dst,
        Imf::Array2D<Imf::Rgba> const& src,
        size_t                         width,
        size_t                         height
        )
{
    auto pitch = width * sizeof(vec3);
    dst.resize(pitch * height);

    vec3* arr = reinterpret_cast<vec3*>(dst.data());

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            arr[y * width + x].x = static_cast<float>(src[y][x].r);
            arr[y * width + x].y = static_cast<float>(src[y][x].g);
            arr[y * width + x].z = static_cast<float>(src[y][x].b);
        }
    }
}

static void store_rgba32f(
        aligned_vector<uint8_t>&       dst,
        Imf::Array2D<Imf::Rgba> const& src,
        size_t                         width,
        size_t                         height
        )
{
    auto pitch = width * sizeof(float);
    dst.resize(pitch * height);

    vec4* arr = reinterpret_cast<vec4*>(dst.data());

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            arr[y * width + x].x = static_cast<float>(src[y][x].r);
            arr[y * width + x].y = static_cast<float>(src[y][x].g);
            arr[y * width + x].z = static_cast<float>(src[y][x].b);
            arr[y * width + x].w = static_cast<float>(src[y][x].a);
        }
    }
}
#endif // VSNRAY_COMMON_HAVE_OPENEXR

bool exr_image::load(std::string const& filename)
{
#if VSNRAY_COMMON_HAVE_OPENEXR
    try
    {
        Imf::RgbaInputFile file(filename.c_str());

        Imath::Box2i dw = file.dataWindow();
        width_ = dw.max.x - dw.min.x + 1;
        height_ = dw.max.y - dw.min.y + 1;

        Imf::Array2D<Imf::Rgba> pixels(height_, width_);

        file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width_, 1, width_);
        file.readPixels(dw.min.y, dw.max.y);

        // Write as 32-bit floats (TODO: have a Visionaray half type?)
        if (file.channels() == Imf::WRITE_RGBA)
        {
            format_ = PF_RGBA32F;

            store_rgba32f(data_, pixels, width_, height_);
        }
        else if (file.channels() == Imf::WRITE_RGB)
        {
            format_ = PF_RGB32F;

            store_rgb32f(data_, pixels, width_, height_);
        }
        else
        {
            std::cerr << "Error: unsupported pixel format\n";
        }

        return true;
    }
    catch(Iex::BaseExc& e)
    {
        std::cerr << "Error: " << e.what() << '\n';

        return false;
    }

    return false;
#else
    VSNRAY_UNUSED(filename);

    return false;
#endif
}

} // visionaray
