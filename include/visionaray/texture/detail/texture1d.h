// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_TEXTURE1D_H
#define VSNRAY_TEXTURE_DETAIL_TEXTURE1D_H 1

#include <cstddef>

#include "texture_common.h"


namespace visionaray
{

template <typename Base, typename T>
class texture_iface<Base, T, 1> : public Base
{
public:

    using ref_type = texture_ref<T, 1>;

    using base_type = Base;
    using value_type = T;

public:

    texture_iface() = default;

    explicit texture_iface(size_t w)
        : Base(w)
        , width_(w)
    {
    }

    template <typename B2>
    texture_iface(texture_iface<B2, T, 1> const& rhs)
        : Base(rhs)
        , width_(rhs.width())
    {
    }

    value_type& operator()(size_t x)
    {
        return base_type::data()[x];
    }

    value_type const& operator()(size_t x) const
    {
        return base_type::data()[x];
    }


    size_t size() const { return width_; }

    size_t width() const { return width_; }

    operator bool() const
    {
        return static_cast<bool>(static_cast<Base>(*this)) && width_ > 0;
    }

private:

    size_t width_;

};

} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_TEXTURE1D_H
