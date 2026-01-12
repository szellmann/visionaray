// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HIP_ARRAY_H
#define VSNRAY_HIP_ARRAY_H 1

#include <cstddef>
#include <cstring> // memset

#include <hip/hip_runtime_api.h>

#include "safe_call.h"

namespace visionaray
{
namespace hip
{

//-------------------------------------------------------------------------------------------------
//
//

class array
{
public:

    // width and height are always *elements*


    array() = default;

    array(hipChannelFormatDesc const& desc, size_t width, size_t height = 0, unsigned flags = 0)
    {
        HIP_SAFE_CALL(allocate(desc, width, height, flags));
    }

    array(hipChannelFormatDesc const& desc, size_t width, size_t height, size_t depth, unsigned flags)
    {
        HIP_SAFE_CALL(allocate3D(desc, width, height, depth, flags));
    }

    array(array&& rhs)
        : array_ptr_(rhs.release())
    {
    }

   ~array()
    {
        reset();
    }

    array& operator=(array&& rhs)
    {
        reset( rhs.release() );
        return *this;
    }


    // NOT copyable
    array(array const& rhs) = delete;
    array& operator=(array const& rhs) = delete;


    hipArray_t get() const
    {
        return array_ptr_;
    }


    hipError_t allocate(hipChannelFormatDesc const& desc, size_t width, size_t height = 0, unsigned flags = 0)
    {
        HIP_SAFE_CALL(hipFree(array_ptr_));

        auto err = hipMallocArray(
                &array_ptr_,
                &desc,
                width,
                height,
                flags
                );

        if (err != hipSuccess)
        {
            array_ptr_ = nullptr;
        }

        return err;
    }

    hipError_t allocate3D(hipChannelFormatDesc const& desc, size_t width, size_t height, size_t depth, unsigned int flags = 0)
    {
        HIP_SAFE_CALL(hipFree(array_ptr_));

        hipExtent extent { width, height, depth };

        auto err = hipMalloc3DArray(&array_ptr_, &desc, extent, flags);

        if (err != hipSuccess)
        {
            array_ptr_ = nullptr;
        }

        return err;
    }

    template <typename T>
    hipError_t upload(T const* host_data, size_t count)
    {
        return hipMemcpyToArray(
                array_ptr_,
                0,
                0,
                host_data,
                count,
                hipMemcpyHostToDevice
                );
    }

    template <typename T>
    hipError_t upload(T const* host_data, size_t width, size_t height, size_t depth)
    {
        hipMemcpy3DParms copy_params;
        memset(&copy_params, 0, sizeof(copy_params));
        copy_params.srcPtr      = make_hipPitchedPtr(
                const_cast<T*>(host_data),
                width * sizeof(T),
                width,
                height
                );
        copy_params.dstArray    = array_ptr_;
        copy_params.extent      = { width, height, depth };
        copy_params.kind        = hipMemcpyHostToDevice;

        return hipMemcpy3D(&copy_params);
    }

private:

    hipArray_t array_ptr_  = nullptr;

    hipArray_t release()
    {
        hipArray_t ptr = array_ptr_;
        array_ptr_ = nullptr;
        return ptr;
    }

    void reset(hipArray_t ptr = nullptr)
    {
        if (array_ptr_)
        {
            HIP_SAFE_CALL(hipFree(array_ptr_));
        }

        array_ptr_ = ptr;
    }

};

} // hip
} // visionaray

#endif // VSNRAY_HIP_ARRAY_H
