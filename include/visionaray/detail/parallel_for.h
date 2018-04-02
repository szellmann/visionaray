// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_PARALLEL_FOR_H
#define VSNRAY_DETAIL_PARALLEL_FOR_H 1

#include <algorithm>
#include <type_traits>

#include "../math/detail/math.h"
#include "thread_pool.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Simple 1-D range class
//

template <typename I>
class range1d
{
public:
    static_assert(std::is_integral<I>::value, "Type must be integral.");

    range1d(I b, I e)
        : begin_(b)
        , end_(e)
    {
    }

    I&        begin()       { return begin_; }
    I const&  begin() const { return begin_; }
    I const& cbegin() const { return begin_; }

    I&        end()         { return end_; }
    I const&  end() const   { return end_; }
    I const& cend() const   { return end_; }

    I length() const
    {
        return end_ - begin_;
    }

private:

    I begin_;
    I end_;
};


//-------------------------------------------------------------------------------------------------
// Simple 2-D range class
//

template <typename I>
class range2d
{
public:
    static_assert(std::is_integral<I>::value, "Type must be integral.");

    range2d(I rb, I re, I cb, I ce)
        : rows_(rb, re)
        , cols_(cb, ce)
    {
    }

    range1d<I>& rows()             { return rows_; }
    range1d<I> const& rows() const { return rows_; }

    range1d<I>& cols()             { return cols_; }
    range1d<I> const& cols() const { return cols_; }

private:

    range1d<I> rows_;
    range1d<I> cols_;
};


//-------------------------------------------------------------------------------------------------
// 1-D tiled range class
//

template <typename I>
class tiled_range1d
{
public:
    static_assert(std::is_integral<I>::value, "Type must be integral.");

    tiled_range1d(I b, I e, I ts)
        : begin_(b)
        , end_(e)
        , tile_size_(ts)
    {
    }

    I&        begin()           { return begin_; }
    I const&  begin()     const { return begin_; }
    I const& cbegin()     const { return begin_; }

    I&        end()             { return end_; }
    I const&  end()       const { return end_; }
    I const& cend()       const { return end_; }

    I&        tile_size()       { return tile_size_; }
    I const&  tile_size() const { return tile_size_; }

    I length() const
    {
        return end_ - begin_;
    }

private:

    I begin_;
    I end_;
    I tile_size_;
};


//-------------------------------------------------------------------------------------------------
// 2-D tiled range class
//

template <typename I>
class tiled_range2d
{
public:
    static_assert(std::is_integral<I>::value, "Type must be integral.");

    tiled_range2d(I rb, I re, I rts, I cb, I ce, I cts)
        : rows_(rb, re, rts)
        , cols_(cb, ce, cts)
    {
    }

    tiled_range1d<I>& rows()             { return rows_; }
    tiled_range1d<I> const& rows() const { return rows_; }

    tiled_range1d<I>& cols()             { return cols_; }
    tiled_range1d<I> const& cols() const { return cols_; }

private:

    tiled_range1d<I> rows_;
    tiled_range1d<I> cols_;
};


//-------------------------------------------------------------------------------------------------
// parallel_for
//

template <typename I, typename Func>
void parallel_for(thread_pool& pool, range1d<I> const& range, Func const& func)
{
    I len = range.length();
    I tile_size = div_up(len, static_cast<I>(pool.num_threads));
    I num_tiles = div_up(len, tile_size);

    pool.run([=](long tile_index)
        {
            I first = static_cast<I>(tile_index) * tile_size;
            I last = std::min(first + tile_size, len);

            for (I i = first; i != last; ++i)
            {
                func(i);
            }

        }, static_cast<long>(num_tiles));
}

template <typename I, typename Func>
void parallel_for(thread_pool& pool, tiled_range1d<I> const& range, Func const& func)
{
    I beg = range.begin();
    I len = range.length();
    I tile_size = range.tile_size();
    I num_tiles = div_up(len, tile_size);

    pool.run([=](long tile_index)
        {
            I first = static_cast<I>(tile_index) * tile_size + beg;
            I last = std::min(first + tile_size, beg + len);

            func(range1d<I>(first, last));

        }, static_cast<long>(num_tiles));
}

template <typename I, typename Func>
void parallel_for(thread_pool& pool, tiled_range2d<I> const& range, Func const& func)
{
    I first_row = range.rows().begin();
    I first_col  = range.cols().begin();
    I width = range.rows().length();
    I height = range.cols().length();
    I tile_width = range.rows().tile_size();
    I tile_height = range.cols().tile_size();
    I num_tiles_x = div_up(width, tile_width);
    I num_tiles_y = div_up(height, tile_height);

    pool.run([=](long tile_index)
        {
            I first_x = (tile_index % num_tiles_x) * tile_width + first_row;
            I last_x = std::min(first_x + tile_width, first_row + width);

            I first_y = (tile_index / num_tiles_x) * tile_height + first_col;
            I last_y = std::min(first_y + tile_height, first_col + height);

            func(range2d<I>(first_x, last_x, first_y, last_y));

        }, static_cast<long>(num_tiles_x * num_tiles_y));
}

} // visionaray

#endif // VSNRAY_DETAIL_PARALLEL_FOR_H
