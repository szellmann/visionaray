// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_THREAD_POOL_H
#define VSNRAY_DETAIL_THREAD_POOL_H 1

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "semaphore.h"

namespace visionaray
{

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
            threads[i] = std::thread([this](){ thread_loop(); });
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

    template <typename Task>
    void run(Task t, long queue_length)
    {
        // Set worker function
        task = t;

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

    // Work-item specific tasks
    using task_t = std::function<void(unsigned)>;
    task_t task;


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

    void thread_loop()
    {
        for (;;)
        {
            // Wait until activated
            {
                std::unique_lock<std::mutex> lock(sync_params.mutex);
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

                task(work_item);

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

} // visionaray

#endif // VSNRAY_DETAIL_THREAD_POOL_H
