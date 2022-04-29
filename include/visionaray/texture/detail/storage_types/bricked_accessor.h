// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_BRICKED_ACCESSOR_H
#define VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_BRICKED_ACCESSOR_H 1

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

#include "../../../math/detail/math.h"
#include "../../../math/simd/gather.h"
#include "../../../math/simd/type_traits.h"
#include "../../../aligned_vector.h"
#include "../../../pixel_format.h"
#include "../../../swizzle.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Simple linear storage type. Data is aligned to allow for SIMD access
//

template <typename T>
class bricked_accessor
{
public:

    using value_type = T;

    static constexpr unsigned BW = 4;
    static constexpr unsigned BH = 4;
    static constexpr unsigned BD = 4;

public:

    bricked_accessor() = default;

    explicit bricked_accessor(std::array<unsigned, 3> size)
        : data_(nullptr)
        , size_(size)
    {
        rounded_size_[0] = div_up(size_[0], BW) * BW;
        rounded_size_[1] = div_up(size_[1], BH) * BH;
        rounded_size_[2] = div_up(size_[2], BD) * BD;

        num_bricks_[0] = div_up(size_[0], BW);
        num_bricks_[1] = div_up(size_[1], BH);
        num_bricks_[2] = div_up(size_[2], BD);
    }

    explicit bricked_accessor(T const* data, std::array<unsigned, 3> size)
        : data_(data)
        , size_(size)
    {
        rounded_size_[0] = div_up(size_[0], BW) * BW;
        rounded_size_[1] = div_up(size_[1], BH) * BH;
        rounded_size_[2] = div_up(size_[2], BD) * BD;

        num_bricks_[0] = div_up(size_[0], BW);
        num_bricks_[1] = div_up(size_[1], BH);
        num_bricks_[2] = div_up(size_[2], BD);
    }

    explicit bricked_accessor(unsigned w, unsigned h, unsigned d)
    {
        size_[0] = w;
        size_[1] = h;
        size_[2] = d;

        rounded_size_[0] = div_up(size_[0], BW) * BW;
        rounded_size_[1] = div_up(size_[1], BH) * BH;
        rounded_size_[2] = div_up(size_[2], BD) * BD;

        num_bricks_[0] = div_up(size_[0], BW);
        num_bricks_[1] = div_up(size_[1], BH);
        num_bricks_[2] = div_up(size_[2], BD);
    }

    std::array<unsigned, 3> size() const
    {
        return size_;
    }

    template <typename U, typename I>
    U value(U /* */, I const& x, I const& y, I const& z) const
    {
        I bx = x / BW;
        I by = y / BH;
        I bz = z / BD;

        I brick_id = bz * num_bricks_[0] * num_bricks_[1]
                   + by * num_bricks_[0]
                   + bx;
        I brick_offset = brick_id * BW * BH * BD;

        I ix = x % BW;
        I iy = y % BH;
        I iz = z % BD;

        I index = brick_offset + iz * BW * BH + iy * BW + ix;

        return access(U{}, index);
    }

    void reset(T const* data)
    {
        data_ = data;
    }

    value_type const* data() const
    {
        return data_;
    }

    operator bool() const
    {
        return data_ != nullptr;
    }

protected:

    template <typename U, typename I>
    U access(U /* */, I const& index) const
    {
        return U(data_[index]);
    }

    template <
        typename U,
        typename I,
        typename = typename std::enable_if<simd::is_simd_vector<I>::value>::type
        >
    U access(U /* */, I const& index) const
    {
        return U(gather(data_, index));
    }

    T const* data_ = nullptr;
    std::array<unsigned, 3> size_;
    std::array<unsigned, 3> rounded_size_;
    std::array<unsigned, 3> num_bricks_;

};

} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_BRICKED_ACCESSOR_H
