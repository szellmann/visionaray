// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cuda_runtime_api.h>

namespace visionaray
{
namespace cuda
{

class texture_object
{
public:

    texture_object() = default;

    explicit texture_object(cudaTextureObject_t obj)
        : texture_object_(obj)
    {
    }

    texture_object(texture_object&& rhs)
        : texture_object_(rhs.release())
    {
    }

   ~texture_object()
    {
        reset();
    }

    texture_object& operator=(texture_object&& rhs)
    {
        reset( rhs.release() );
        return *this;
    }


    // NOT copyable
    texture_object(texture_object const& rhs) = delete;
    texture_object& operator=(texture_object const& rhs) = delete;



    cudaTextureObject_t get() const
    {
        return texture_object_;
    }

    void reset(cudaTextureObject_t obj = 0)
    {
        if (texture_object_)
        {
            cudaDestroyTextureObject( texture_object_ );
        }

        texture_object_ = obj;
    }

    cudaTextureObject_t release()
    {
        auto tmp = texture_object_;
        texture_object_ = 0;
        return tmp;
    }

private:

    cudaTextureObject_t texture_object_ = 0;

};

} // cuda
} // visionaray
