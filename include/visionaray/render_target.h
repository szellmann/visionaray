// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RENDER_TARGET_H
#define VSNRAY_RENDER_TARGET_H

#include <cstddef>

namespace visionaray
{

class render_target
{
public:

    void resize(size_t w, size_t h)
    {
        width_  = w;
        height_ = h;
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }

private:

    size_t width_;
    size_t height_;

};

} // visionaray

#endif // VSNRAY_RENDER_TARGET_H
