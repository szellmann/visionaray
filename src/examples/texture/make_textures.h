
#pragma once

#ifndef VSNRAY_TEXTURE_EX_MAKE_TEXTURES_H
#define VSNRAY_TEXTURE_EX_MAKE_TEXTURES_H 1

#include <complex>

namespace texture_ex
{

std::vector<uint8_t> make_mandel()
{
    int w = 80;
    int h = 60;

    double cxmin = -40;
    double cxmax =  40;
    double cymin = -40;
    double cymax =  40;

    int iterations = 100;

    std::vector<uint8_t> result(w * h * 3);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            std::complex<double> c(
                cxmin + x / (w - 1.0) * (cxmax - cxmin),
                cymin + y / (h - 1.0) * (cymax - cymin)
                );
            std::complex<double> z = 0;
         
            for (int i = 0; i < iterations && std::abs(z) < 2.0; ++i)
            {
                z = z * z + c;

                result[(y * w + h) * 3 + 0] = (i == iterations) ? 255 : 0;//set_color : non_set_color;
                result[(y * w + h) * 3 + 1] = (i == iterations) ? 255 : 0;
                result[(y * w + h) * 3 + 2] = (i == iterations) ? 255 : 0;
            }
        }
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Create an 8x1 RGB8 rainbow texture
//

std::vector<uint8_t> make_rainbow()
{
    std::vector<uint8_t> result(8 * 1 * 3);

    result[0] = 255;
    result[1] = 0;
    result[2] = 0;

    result[3] = 255;
    result[4] = 127;
    result[5] = 0;

    result[6] = 255;
    result[7] = 255;
    result[8] = 0;

    result[9] = 0;
    result[10] = 255;
    result[11] = 0;

    result[12] = 0;
    result[13] = 255;
    result[14] = 255;

    result[15] = 0;
    result[16] = 0;
    result[17] = 255;

    result[18] = 128;
    result[19] = 0;
    result[20] = 255;

    result[21] = 255;
    result[22] = 0;
    result[23] = 255;

    return result;
}


//-------------------------------------------------------------------------------------------------
// Create a 16x16 checkerboard texture
//

std::vector<uint8_t> make_checkerboard()
{
    int w = 16;
    int h = 16;

    std::vector<uint8_t> result(w * h * 3);

    uint8_t val = 255;
    for (int y = 0; y < h; ++y)
    {
        val = val == 0 ? 255 : 0;       // only works if w and h even!
        for (int x = 0; x < w; ++x)
        {
            val = val == 0 ? 255 : 0;   // only works if w and h even!
            result[(y * w + x) * 3    ] = val;
            result[(y * w + x) * 3 + 1] = val;
            result[(y * w + x) * 3 + 2] = val;
        }
    }

    return result;
}

} // texture_ex

#endif // VSNRAY_TEXTURE_EX_MAKE_TEXTURES_H
