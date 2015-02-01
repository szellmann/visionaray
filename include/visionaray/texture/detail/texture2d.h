// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_TEXTURE2D_H
#define VSNRAY_TEXTURE_TEXTURE2D_H

#include <cstddef>

#include "texture_common.h"


namespace visionaray
{


template
<
    typename T,
    tex_read_mode ReadMode
>
class texture<T, ReadMode, 2> : public texture_storage<T, texture<T, ReadMode, 2>>,
    public prefilterable<T, texture<T, ReadMode, 2>>
{
public:

    typedef texture_storage<T, texture> storage_base;
    typedef T value_type;


    texture() {}

    texture(size_t w, size_t h)
        : width_(w)
        , height_(h)
    {
    }


    value_type& operator()(size_t x, size_t y)
    {
        return storage_base::data[y * width_ + x];
    }

    value_type const& operator()(size_t x, size_t y) const
    {
        return storage_base::data[y * width_ + x];
    }


    size_t size() const { return width_ * height_; }

    size_t width() const { return width_; }
    size_t height() const { return height_; }

    void resize(size_t w, size_t h)
    {
        width_ = w;
        height_ = h;
    }

private:

    size_t width_;
    size_t height_;

};


} // visionaray


#endif // VSNRAY_TEXTURE_TEXTURE2D_H


