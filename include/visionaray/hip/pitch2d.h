// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HIP_PITCH2D_H
#define VSNRAY_HIP_PITCH2D_H 1

#include <cstddef>

#include <hip/hip_runtime_api.h>

#include "safe_call.h"

namespace visionaray
{
namespace hip
{

//-------------------------------------------------------------------------------------------------
//
//

template <typename T>
class pitch2d
{
public:

    // width and height are always *elements*


    pitch2d() {}

    pitch2d(size_t width, size_t height)
    {
        HIP_SAFE_CALL(allocate(width, height));
    }

    pitch2d(pitch2d&& rhs)
        : device_ptr_(rhs.release())
        , dpitch_(rhs.dpitch_)
    {
    }

   ~pitch2d()
    {
        reset();
    }

    pitch2d& operator=(pitch2d&& rhs)
    {
        reset( rhs.release() );
        dpitch_ = rhs.dpitch_;
        return *this;
    }


    // NOT copyable
    pitch2d(pitch2d const& rhs) = delete;
    pitch2d& operator=(pitch2d const& rhs) = delete;


    T* get()
    {
        return device_ptr_;
    }

    T const* get() const
    {
        return device_ptr_;
    }

    size_t get_pitch_in_bytes() const
    {
        return dpitch_;
    }


    hipError_t allocate(size_t width, size_t height)
    {
        HIP_SAFE_CALL(hipFree(device_ptr_));

        auto err = hipMallocPitch(
                (void**)&device_ptr_,
                &dpitch_,
                sizeof(T) * width,
                height
                );

        if (err != hipSuccess)
        {
            device_ptr_ = nullptr;
        }

        return err;
    }

    hipError_t upload(T const* host_data, size_t width, size_t height)
    {
        return hipMemcpy2D(
                device_ptr_,
                dpitch_,
                host_data,
                sizeof(T) * width,
                sizeof(T) * width,
                height,
                hipMemcpyDefault
                );
    }

private:

    T* device_ptr_  = nullptr;
    size_t dpitch_  = 0;

    T* release()
    {
        T* ptr = device_ptr_;
        device_ptr_ = nullptr;
        return ptr;
    }

    void reset(T* ptr = nullptr)
    {
        if (device_ptr_)
        {
            HIP_SAFE_CALL(hipFree(device_ptr_));
        }

        device_ptr_ = ptr;
    }

};

} // hip
} // visionaray

#endif // VSNRAY_HIP_PITCH2D_H
