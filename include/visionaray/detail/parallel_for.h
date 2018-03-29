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
// Simple range class
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

    I&        end()         { return end; }
    I const&  end() const   { return end; }
    I const& cend() const   { return end; }

    I length() const
    {
        return end_ - begin_;
    }

private:

    I begin_;
    I end_;
};


//-------------------------------------------------------------------------------------------------
// Tiled range class
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
        return (end_ - begin_) * tile_size_;
    }   

private:

    I begin_;
    I end_;
    I tile_size_;
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
            unsigned first = static_cast<unsigned>(tile_index) * tile_size;
            unsigned last = std::min(first + tile_size, len);

            for (unsigned i = first; i != last; ++i)
            {
                func(i);
            }

        }, static_cast<long>(num_tiles));
}

} // visionaray

#endif // VSNRAY_DETAIL_PARALLEL_FOR_H
