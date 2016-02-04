// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_TEXTURE2D_H
#define VSNRAY_TEXTURE_TEXTURE2D_H 1

#include <cstddef>
#include <type_traits>

#include "texture_common.h"


namespace visionaray
{

template <typename Base, typename T, tex_read_mode ReadMode>
class texture_iface<Base, T, ReadMode, 2> : public Base
{
public:

    using base_type = Base;
    using value_type = T;

#ifdef VSNRAY_CXX_HAS_INHERITING_CONSTRUCTORS
    using Base::Base;
#endif

public:

    texture_iface() = default;

    texture_iface(size_t w, size_t h)
        : Base(w * h)
        , width_(w)
        , height_(h)
    {
    }

    template <typename B2>
    texture_iface(texture_iface<B2, T, ReadMode, 2> const& rhs)
        : Base(rhs)
        , width_(rhs.width())
        , height_(rhs.height())
    {
    }

#ifndef VSNRAY_CXX_HAS_INHERITING_CONSTRUCTORS
    template <
        typename A,
        typename ...Args,
        typename = typename std::enable_if<std::is_constructible<Base, A, Args...>::value>::type
        >
    texture_iface(A&& a, Args&&... args)
        : Base(std::forward<A>(a), std::forward<Args>(args)...)
    {
    }
#endif

    value_type& operator()(size_t x, size_t y)
    {
        return base_type::data()[y * width_ + x];
    }

    value_type const& operator()(size_t x, size_t y) const
    {
        return base_type::data()[y * width_ + x];
    }


    vector<2, size_t> size() const
    {
        return vector<2, size_t>(width_, height_);
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }

    void resize(size_t w, size_t h)
    {
        width_ = w;
        height_ = h;
        Base::reset(w * h);
    }

private:

    size_t width_;
    size_t height_;

};

} // visionaray

#endif // VSNRAY_TEXTURE_TEXTURE2D_H
