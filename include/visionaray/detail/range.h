// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_RANGE_FOR_H
#define VSNRAY_DETAIL_RANGE_FOR_H 1

#include "macros.h"

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

    VSNRAY_FUNC range1d(I b, I e)
        : begin_(b)
        , end_(e)
    {
    }

    VSNRAY_FUNC I&        begin()       { return begin_; }
    VSNRAY_FUNC I const&  begin() const { return begin_; }
    VSNRAY_FUNC I const& cbegin() const { return begin_; }

    VSNRAY_FUNC I&        end()         { return end_; }
    VSNRAY_FUNC I const&  end() const   { return end_; }
    VSNRAY_FUNC I const& cend() const   { return end_; }

    VSNRAY_FUNC I length() const
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

    VSNRAY_FUNC range2d(I rb, I re, I cb, I ce)
        : rows_(rb, re)
        , cols_(cb, ce)
    {
    }

    VSNRAY_FUNC range1d<I>& rows()             { return rows_; }
    VSNRAY_FUNC range1d<I> const& rows() const { return rows_; }

    VSNRAY_FUNC range1d<I>& cols()             { return cols_; }
    VSNRAY_FUNC range1d<I> const& cols() const { return cols_; }

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

    VSNRAY_FUNC tiled_range1d(I b, I e, I ts)
        : begin_(b)
        , end_(e)
        , tile_size_(ts)
    {
    }

    VSNRAY_FUNC I&        begin()           { return begin_; }
    VSNRAY_FUNC I const&  begin()     const { return begin_; }
    VSNRAY_FUNC I const& cbegin()     const { return begin_; }

    VSNRAY_FUNC I&        end()             { return end_; }
    VSNRAY_FUNC I const&  end()       const { return end_; }
    VSNRAY_FUNC I const& cend()       const { return end_; }

    VSNRAY_FUNC I&        tile_size()       { return tile_size_; }
    VSNRAY_FUNC I const&  tile_size() const { return tile_size_; }

    VSNRAY_FUNC I length() const
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

    VSNRAY_FUNC tiled_range2d(I rb, I re, I rts, I cb, I ce, I cts)
        : rows_(rb, re, rts)
        , cols_(cb, ce, cts)
    {
    }

    VSNRAY_FUNC tiled_range1d<I>& rows()             { return rows_; }
    VSNRAY_FUNC tiled_range1d<I> const& rows() const { return rows_; }

    VSNRAY_FUNC tiled_range1d<I>& cols()             { return cols_; }
    VSNRAY_FUNC tiled_range1d<I> const& cols() const { return cols_; }

private:

    tiled_range1d<I> rows_;
    tiled_range1d<I> cols_;
};

} // visionaray

#endif // VSNRAY_DETAIL_RANGE_FOR_H
