// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_BRICKED_STORAGE_H
#define VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_BRICKED_STORAGE_H 1

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

template <typename T, size_t A = 16>
class bricked_storage
{
public:

    using value_type = T;

    static constexpr unsigned BW = 4;
    static constexpr unsigned BH = 4;
    static constexpr unsigned BD = 4;

public:

    bricked_storage() = default;

    explicit bricked_storage(std::array<unsigned, 3> size)
    {
        realloc(size[0], size[1], size[2]);
    }

    explicit bricked_storage(unsigned w, unsigned h, unsigned d)
    {
        realloc(w, h, d);
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

    void realloc(unsigned w, unsigned h, unsigned d)
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

        data_.resize(rounded_size_[0] * rounded_size_[1] * rounded_size_[2]);
    }

    void reset(T const* data)
    {
        for (unsigned bz = 0; bz < num_bricks_[2]; ++bz)
        {
            for (unsigned by = 0; by < num_bricks_[1]; ++by)
            {
                for (unsigned bx = 0; bx < num_bricks_[0]; ++bx)
                {
                    for (unsigned iz = 0; iz < BD; ++iz)
                    {
                        for (unsigned iy = 0; iy < BH; ++iy)
                        {
                            for (unsigned ix = 0; ix < BW; ++ix)
                            {
                                unsigned x = bx * BW + ix;
                                unsigned y = by * BH + iy;
                                unsigned z = bz * BD + iz;

                                if (x >= size_[0] || y >= size_[1] || z >= size_[2])
                                {
                                    continue;
                                }

                                size_t brick_id = size_t(bz) * num_bricks_[0] * num_bricks_[1]
                                                        + by * num_bricks_[0]
                                                        + bx;
                                size_t brick_offset = brick_id * BW * BH * BD;
                                size_t dest_index = brick_offset + iz * BW * BH + iy * BW + ix;

                                size_t source_index = size_t(z) * size_[0] * size_[1] + y * size_[0] + x;

                                data_[dest_index] = data[source_index];
                            }
                        }
                    }
                }
            }
        }
    }

    void reset(
            T const* data,
            pixel_format format,
            pixel_format internal_format
            )
    {
        if (format != internal_format)
        {
            // Swizzle in-place
            aligned_vector<T> tmp(data, data + data_.size());
            swizzle(tmp.data(), internal_format, format, tmp.size());
            reset(tmp.data());
        }
        else
        {
            // Simple copy
            reset(data);
        }
    }

    template <typename U>
    void reset(
            U const* data,
            pixel_format format,
            pixel_format internal_format
            )
    {
        // Copy to temporary array, then swizzle
        aligned_vector<T> dst(data_.size());
        swizzle(dst.data(), internal_format, data, format, dst.size());
        reset(dst.data());
    }

    template <typename U>
    void reset(
            U const* data,
            pixel_format format,
            pixel_format internal_format,
            swizzle_hint hint
            )
    {
        // Copy with temporary array, hint about how to handle alpha
        aligned_vector<T> dst(data_.size());
        swizzle(dst.data(), internal_format, data, format, dst.size(), hint);
        reset(dst.data());
    }

    value_type const* data() const
    {
        return data_.data();
    }

    operator bool() const
    {
        return !data_.empty();
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

    aligned_vector<T, A> data_;
    std::array<unsigned, 3> size_;
    std::array<unsigned, 3> rounded_size_;
    std::array<unsigned, 3> num_bricks_;

};

} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_STORAGE_TYPES_BRICKED_STORAGE_H
