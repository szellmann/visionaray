// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_SEMAPHORE_H
#define VSNRAY_DETAIL_SEMAPHORE_H

#include "platform.h"

#if defined(VSNRAY_OS_WIN32) || defined(VSNRAY_OS_DARWIN)
#define VSNRAY_DETAIL_SEMAPHORE_USE_STD 1
#endif

#if VSNRAY_DETAIL_SEMAPHORE_USE_STD
#include <condition_variable>
#include <mutex>
#else
#include <semaphore.h>
#endif

namespace visionaray
{

#ifdef VSNRAY_DETAIL_SEMAPHORE_USE_STD

class semaphore
{
public:

    semaphore(unsigned count = 0) : count_(count) {}

    void notify()
    {
        std::unique_lock<std::mutex> l(mutex_);
        ++count_;
        cond_.notify_one();
    }

    void wait()
    {
        std::unique_lock<std::mutex> l(mutex_);
        cond_.wait(l, [this]() { return count_ > 0; });
        count_--;
    }

private:

    std::condition_variable cond_;
    std::mutex mutex_;
    unsigned count_;

};

#else

class semaphore
{
public:

    semaphore(unsigned count = 0)
    {
        sem_init(&sem_, 0, count);
    }

   ~semaphore()
    {
        sem_close(&sem_);
        sem_destroy(&sem_);
    }

    void notify()
    {
        sem_post(&sem_);
    }

    void wait()
    {
        sem_wait(&sem_);
    }

private:

    sem_t sem_;

};

#endif

} // visionaray


#endif // VSNRAY_DETAIL_SEMAPHORE_H


