// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HIP_TEXTURE_OBJECT_H
#define VSNRAY_HIP_TEXTURE_OBJECT_H 1

#include <hip/hip_runtime_api.h>

namespace visionaray
{
namespace hip
{

class texture_object
{
public:

    texture_object() = default;

    explicit texture_object(hipTextureObject_t obj)
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
        reset(rhs.release());
        return *this;
    }


    // NOT copyable
    texture_object(texture_object const& rhs) = delete;
    texture_object& operator=(texture_object const& rhs) = delete;



    hipTextureObject_t get() const
    {
        return texture_object_;
    }

    void reset(hipTextureObject_t obj = 0)
    {
        if (texture_object_)
        {
            hipDestroyTextureObject( texture_object_ );
        }

        texture_object_ = obj;
    }

    hipTextureObject_t release()
    {
        auto tmp = texture_object_;
        texture_object_ = 0;
        return tmp;
    }

private:

    hipTextureObject_t texture_object_ = 0;

};

} // hip
} // visionaray

#endif // VSNRAY_CUDA_TEXTURE_OBJECT_H
