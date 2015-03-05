// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TIMER_H
#define VSNRAY_TIMER_H

#include <chrono>


namespace visionaray
{

class timer
{
public:

    typedef std::chrono::high_resolution_clock clock;
    typedef clock::time_point time_point;
    typedef clock::duration duration;

    timer()
        : start_(clock::now())
    {
    }

    void reset()
    {
        start_ = clock::now();
    }

    double elapsed() const
    {
        return std::chrono::duration<double>(clock::now() - start_).count();
    }

private:

    time_point start_;

};


class frame_counter
{
public:

    frame_counter()
        : count_(0)
    {
    }

    void reset()
    {
        timer_.reset();
        count_ = 0;
        fps_ = 0.0;
    }

    double register_frame()
    {

        ++count_;
        double elapsed = timer_.elapsed();

        if (elapsed > 0.5/*sec*/)
        {
            fps_ = count_ / elapsed;
            timer_.reset();
            count_ = 0;
        }

        return fps_;

    }

private:

    timer timer_;
    unsigned count_;
    double fps_;

};

} // visionaray

#endif // VSNRAY_TIMER_H
