// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_PARALLEL_FOR_H
#define VSNRAY_DETAIL_PARALLEL_FOR_H 1

#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory>
#include <thread>
#include <type_traits>

#include "../math/detail/math.h"
#include "semaphore.h"

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
        : row_begin_(rb)
        , row_end_(re)
        , col_begin_(cb)
        , col_end_(ce)
    {
    }

    I&        row_begin()           { return row_begin_; }
    I const&  row_begin()     const { return row_begin_; }
    I const& crow_begin()     const { return row_begin_; }

    I&        row_end()             { return row_end_; }
    I const&  row_end()       const { return row_end_; }
    I const& crow_end()       const { return row_end_; }

    I&        col_begin()           { return col_begin_; }
    I const&  col_begin()     const { return col_begin_; }
    I const& ccol_begin()     const { return col_begin_; }

    I&        col_end()             { return col_end_; }
    I const&  col_end()       const { return col_end_; }
    I const& ccol_end()       const { return col_end_; }

    I row_length() const
    {
        return row_end_ - row_begin_;
    }

    I col_length() const
    {
        return col_end_ - col_begin_;
    }

private:

    I row_begin_;
    I row_end_;

    I col_begin_;
    I col_end_;
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
        : row_begin_(rb)
        , row_end_(re)
        , row_tile_size_(rts)
        , col_begin_(cb)
        , col_end_(ce)
        , col_tile_size_(cts)
    {
    }

    I&        row_begin()           { return row_begin_; }
    I const&  row_begin()     const { return row_begin_; }
    I const& crow_begin()     const { return row_begin_; }

    I&        row_end()             { return row_end_; }
    I const&  row_end()       const { return row_end_; }
    I const& crow_end()       const { return row_end_; }

    I&        row_tile_size()       { return row_tile_size_; }
    I const&  row_tile_size() const { return row_tile_size_; }

    I&        col_begin()           { return col_begin_; }
    I const&  col_begin()     const { return col_begin_; }
    I const& ccol_begin()     const { return col_begin_; }

    I&        col_end()             { return col_end_; }
    I const&  col_end()       const { return col_end_; }
    I const& ccol_end()       const { return col_end_; }

    I&        col_tile_size()       { return col_tile_size_; }
    I const&  col_tile_size() const { return col_tile_size_; }

    I row_length() const
    {
        return row_end_ - row_begin_;
    }

    I col_length() const
    {
        return col_end_ - col_begin_;
    }

private:

    I row_begin_;
    I row_end_;
    I row_tile_size_;

    I col_begin_;
    I col_end_;
    I col_tile_size_;
};


//-------------------------------------------------------------------------------------------------
// Thread pool
//

class thread_pool
{
public:

    explicit thread_pool(unsigned num_threads)
    {
        sync_params.start_threads = false;
        sync_params.join_threads = false;
        reset(num_threads);
    }

   ~thread_pool()
    {
        join_threads();
    }

    void reset(unsigned num_threads)
    {
        join_threads();

        threads.reset(new std::thread[num_threads]);
        this->num_threads = num_threads;

        for (unsigned i = 0; i < num_threads; ++i)
        {
            threads[i] = std::thread([this,i](){ thread_loop(i); });
        }
    }

    void join_threads()
    {
        if (num_threads == 0)
        {
            return;
        }

        sync_params.start_threads = true;
        sync_params.join_threads = true;
        sync_params.threads_start.notify_all();

        for (unsigned i = 0; i < num_threads; ++i)
        {
            if (threads[i].joinable())
            {
                threads[i].join();
            }
        }

        sync_params.start_threads = false;
        sync_params.join_threads = false;
        threads.reset(nullptr);
    }

    template <typename Func>
    void run(Func f, long queue_length)
    {
        // Set worker function
        func = f;

        // Set counters
        sync_params.num_work_items = queue_length;
        sync_params.work_item_counter = 0;
        sync_params.work_items_finished_counter = 0;

        // Activate persistent threads
        sync_params.start_threads = true;
        sync_params.threads_start.notify_all();

        // Wait for all threads to finish
        sync_params.threads_ready.wait();

        // Idle w/o work
        sync_params.start_threads = false;
    }

    std::unique_ptr<std::thread[]> threads;
    unsigned num_threads = 0;

private:

    using func_t = std::function<void(unsigned)>;
    func_t func;


    struct
    {
        std::mutex              mutex;
        std::condition_variable threads_start;
        visionaray::semaphore   threads_ready;

        std::atomic<bool>       start_threads;
        std::atomic<bool>       join_threads;

        std::atomic<long>       num_work_items;
        std::atomic<long>       work_item_counter;
        std::atomic<long>       work_items_finished_counter;
    } sync_params;

    void thread_loop(unsigned id)
    {
        for (;;)
        {
            // Wait until activated
            {
                std::unique_lock<std::mutex> lock(sync_params.mutex);
                auto const& start_threads = sync_params.start_threads;
                sync_params.threads_start.wait(
                        lock,
                        [this]()
                            -> std::atomic<bool> const&
                        {
                            return sync_params.start_threads;
                        }
                        );
            }

            // Exit?
            if (sync_params.join_threads)
            {
                break;
            }


            // Perform work in queue
            for (;;)
            {
                auto work_item = sync_params.work_item_counter.fetch_add(1);

                if (work_item >= sync_params.num_work_items)
                {
                    break;
                }

                func(work_item);

                auto finished = sync_params.work_items_finished_counter.fetch_add(1);

                if (finished >= sync_params.num_work_items - 1)
                {
                    assert(finished == sync_params.num_work_items - 1);
                    sync_params.threads_ready.notify();
                    break;
                }
            }
        }
    }
};

template <typename I, typename Func>
void parallel_for(thread_pool& pool, range1d<I> const& range, Func const& func)
{
    unsigned len = static_cast<unsigned>(range.length());
    unsigned tile_size = div_up(len, pool.num_threads);
    unsigned num_tiles = div_up(len, tile_size);

    pool.run([&](long tile_index)
        {
            unsigned first = static_cast<unsigned>(tile_index) * tile_size;
            unsigned last = std::min(first + tile_size, len);

            for (unsigned i = first; i != last; ++i)
            {
                func(i);
            }

        }, static_cast<long>(num_tiles));
}

template <typename I, typename Func>
void parallel_for(thread_pool& pool, tiled_range1d<I> const& range, Func const& func)
{
    unsigned len = static_cast<unsigned>(range.length());
    unsigned tile_size = static_cast<unsigned>(range.tile_size());
    unsigned num_tiles = div_up(len, tile_size);

    pool.run([&](long tile_index)
        {
            I first = static_cast<I>(tile_index) * static_cast<I>(tile_size);
            I last = std::min(first + static_cast<I>(tile_size), static_cast<I>(len));

            func(range1d<I>(first, last));

        }, static_cast<long>(num_tiles));
}

template <typename I, typename Func>
void parallel_for(thread_pool& pool, tiled_range2d<I> const& range, Func const& func)
{
    unsigned tile_width = static_cast<unsigned>(range.row_tile_size());
    unsigned tile_height = static_cast<unsigned>(range.col_tile_size());
    unsigned num_tiles_x = div_up(static_cast<unsigned>(range.row_length()), tile_width);
    unsigned num_tiles_y = div_up(static_cast<unsigned>(range.col_length()), tile_height);

    pool.run([&](long tile_index)
        {
            func(range2d<I>(
                    (tile_index % num_tiles_x) * tile_width,
                    tile_width,
                    (tile_index / num_tiles_x) * tile_height,
                    tile_height
                    ));

        }, static_cast<long>(num_tiles_x * num_tiles_y));
}

} // visionaray

#endif // VSNRAY_DETAIL_PARALLEL_FOR_H
