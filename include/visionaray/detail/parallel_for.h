// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_PARALLEL_FOR_H
#define VSNRAY_DETAIL_PARALLEL_FOR_H 1

#include "../math/detail/math.h"
#include "range.h"
#include "thread_pool.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// parallel_for
//

template <typename I, typename Func>
void parallel_for(thread_pool& pool, range1d<I> const& range, Func const& func)
{
    I len = range.length();
    I tile_size = div_up(len, static_cast<I>(pool.num_threads));
    I num_tiles = div_up(len, tile_size);

    pool.run([=](unsigned tile_index)
        {
            I first = static_cast<I>(tile_index) * tile_size;
            I last = min(first + tile_size, len);

            for (I i = first; i != last; ++i)
            {
                func(i);
            }

        }, static_cast<unsigned>(num_tiles));
}

template <typename I, typename Func>
void parallel_for(thread_pool& pool, tiled_range1d<I> const& range, Func const& func)
{
    I beg = range.begin();
    I len = range.length();
    I tile_size = range.tile_size();
    I num_tiles = div_up(len, tile_size);

    pool.run([=](unsigned tile_index)
        {
            I first = static_cast<I>(tile_index) * tile_size + beg;
            I last = min(first + tile_size, beg + len);

            func(range1d<I>(first, last));

        }, static_cast<unsigned>(num_tiles));
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

    pool.run([=](unsigned tile_index)
        {
            I first_x = (tile_index % num_tiles_x) * tile_width + first_row;
            I last_x = min(first_x + tile_width, first_row + width);

            I first_y = (tile_index / num_tiles_x) * tile_height + first_col;
            I last_y = min(first_y + tile_height, first_col + height);

            func(range2d<I>(first_x, last_x, first_y, last_y));

        }, static_cast<unsigned>(num_tiles_x * num_tiles_y));
}

} // visionaray

#endif // VSNRAY_DETAIL_PARALLEL_FOR_H
