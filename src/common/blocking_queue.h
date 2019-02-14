// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_BLOCKING_QUEUE_H
#define VSNRAY_COMMON_BLOCKING_QUEUE_H 1

#include <cassert>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>

#include <visionaray/detail/semaphore.h>

namespace visionaray
{

template <typename T, typename A = std::allocator<T>>
class blocking_queue
{
public:

    using queue_type = std::deque<T>;

public:

    void push_back(T const& value)
    {
        {
            std::unique_lock<std::mutex> l(mutex_);

            queue_.push_back(value);
        }

        non_empty_.notify();
    }

    void push_back(T&& value)
    {
        {
            std::unique_lock<std::mutex> l(mutex_);

            queue_.push_back(std::move(value));
        }

        non_empty_.notify();
    }

    T&& pop_front()
    {
        T next;

        // Wait until an object was appended to the queue
        non_empty_.wait();

        {
            std::unique_lock<std::mutex> l(mutex_);

            assert(!queue_.empty());

            // Return the front of the queue
            next = std::move(queue_.front());
            // Remove the object from the queue
            queue_.pop_front();
        }

        return std::move(next);
    }

private:

    // The queue
    queue_type queue_;

    // Mutex to protect the queue
    std::mutex mutex_;

    // Signaled if the queue becomes non-empty
    semaphore non_empty_;
};

} // visionaray

#endif // VSNRAY_COMMON_BLOCKING_QUEUE_H
