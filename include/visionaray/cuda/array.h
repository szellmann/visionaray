// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_ARRAY_H
#define VSNRAY_CUDA_ARRAY_H 1

#include <cassert>

#include <cuda_runtime_api.h>

#include <visionaray/detail/macros.h>

namespace visionaray
{
namespace cuda
{

//-------------------------------------------------------------------------------------------------
//
//

class array
{
public:

    // width and height are always *elements*


    array() = default;

    array(cudaChannelFormatDesc const& desc, size_t width, size_t height = 0, unsigned flags = 0)
    {
        allocate(desc, width, height, flags);
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


    cudaArray_t get()
    {
        return array_ptr_;
    }


    cudaError_t allocate(cudaChannelFormatDesc const& desc, size_t width, size_t height = 0, unsigned flags = 0)
    {
        cudaFree(array_ptr_);

        auto err = cudaMallocArray(
                &array_ptr_,
                &desc,
                width,
                height,
                flags
                );

        if (err != cudaSuccess)
        {
            array_ptr_ = nullptr;
        }

        return err;
    }

    template <typename T>
    cudaError_t upload(T const* host_data, size_t count)
    {
        return cudaMemcpyToArray(
                array_ptr_,
                0,
                0,
                host_data,
                count,
                cudaMemcpyHostToDevice
                );
    }

private:

    cudaArray_t array_ptr_  = nullptr;

    cudaArray_t release()
    {
        cudaArray_t ptr = array_ptr_;
        array_ptr_ = nullptr;
        return ptr;
    }

    void reset(cudaArray_t ptr = nullptr)
    {
        if (array_ptr_)
        {
            cudaFree(array_ptr_);
        }

        array_ptr_ = ptr;
    }

};


} // cuda
} // visionaray

#endif // VSNRAY_CUDA_ARRAY_H
