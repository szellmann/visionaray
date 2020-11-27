// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_TIMER_H
#define VSNRAY_COMMON_TIMER_H 1

#include <chrono>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif


namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Timer class using std::chrono's high-resolution clock
//

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


#ifdef __CUDACC__

namespace cuda
{

//-------------------------------------------------------------------------------------------------
// CUDA event-based timer class
//

class timer
{
public:

    timer()
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);

        reset();
    }

    ~timer()
    {
        cudaEventDestroy(stop_);
        cudaEventDestroy(start_);
    }

    void reset()
    {
        cudaEventRecord(start_);
    }

    double elapsed() const
    {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return static_cast<double>(ms) / 1000.0;
    }

private:

    cudaEvent_t start_;
    cudaEvent_t stop_;

};

} // cuda

#endif



//-------------------------------------------------------------------------------------------------
// basic_frame_counter
//

template <typename Timer>
class basic_frame_counter
{
public:

    basic_frame_counter()
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

    Timer timer_;
    unsigned count_;
    double fps_;

};


//-------------------------------------------------------------------------------------------------
// frame counter platform typedefs
//

using frame_counter = basic_frame_counter<timer>;

#ifdef __CUDACC__
namespace cuda
{
using frame_counter = basic_frame_counter<cuda::timer>;
}
#endif

} // visionaray

#endif // VSNRAY_COMMON_TIMER_H
