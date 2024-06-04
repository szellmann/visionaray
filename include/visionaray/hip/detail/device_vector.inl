// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <hip/hip_runtime.h>

#include "../fill.h"
#include "../safe_call.h"

namespace visionaray
{
namespace hip
{

template <typename T>
device_vector<T>::~device_vector()
{
    HIP_SAFE_CALL(hipFree(data_));
}

template <typename T>
device_vector<T>::device_vector(device_vector<T> const& rhs)
    : size_(rhs.size())
{
    if (&rhs != this)
    {
        HIP_SAFE_CALL(hipMalloc(&data_, sizeof(T) * size_));
        HIP_SAFE_CALL(hipMemcpy(
            data_,
            rhs.data(),
            sizeof(T) * size_,
            hipMemcpyDeviceToDevice
            ));
        HIP_SAFE_CALL(hipDeviceSynchronize());
    }
}

template <typename T>
device_vector<T>::device_vector(device_vector<T>&& rhs)
{
    if (&rhs != this)
    {
        data_ = std::move(rhs.data_);
        size_ = rhs.size_;

        rhs.data_ = nullptr;
        rhs.size_ = 0;
    }
}

template <typename T>
device_vector<T>::device_vector(size_t size)
    : size_(size)
{
    HIP_SAFE_CALL(hipMalloc(&data_, sizeof(T) * size_));
}

template <typename T>
device_vector<T>::device_vector(size_t size, T const& value)
    : size_(size)
{
    HIP_SAFE_CALL(hipMalloc(&data_, sizeof(T) * size_));
    hip::fill(data_, size_ * sizeof(T), (T*)&value, sizeof(value));
}

template <typename T>
device_vector<T>::device_vector(host_vector<T> const &hv)
    : size_(hv.size())
{
    HIP_SAFE_CALL(hipMalloc(&data_, sizeof(T) * size_));
    HIP_SAFE_CALL(hipMemcpy(
        data_,
        hv.data(),
        sizeof(T) * size_,
        hipMemcpyHostToDevice
        ));
}

template <typename T>
device_vector<T>::device_vector(const T* data, size_t size)
    : size_(size)
{
    HIP_SAFE_CALL(hipMalloc(&data_, sizeof(T) * size_));
    HIP_SAFE_CALL(hipMemcpy(
        data_,
        data,
        sizeof(T) * size_,
        hipMemcpyDefault
        ));
}

template <typename T>
device_vector<T>::device_vector(const T* begin, const T* end)
    : size_(end - begin)
{
    HIP_SAFE_CALL(hipMalloc(&data_, sizeof(T) * size_));
    HIP_SAFE_CALL(hipMemcpy(
        data_,
        begin,
        sizeof(T) * size_,
        hipMemcpyDefault
        ));

  hipPointerAttribute_t attributes;
  HIP_SAFE_CALL(hipPointerGetAttributes(&attributes, begin));

  if (attributes.type == hipMemoryTypeDevice)
  {
      HIP_SAFE_CALL(hipDeviceSynchronize());
  }
}

template <typename T>
device_vector<T>& device_vector<T>::operator=(device_vector<T> const& rhs)
{
    if (&rhs != this)
    {
        size_ = rhs.size_;
        HIP_SAFE_CALL(hipMalloc(&data_, sizeof(T) * size_));
        HIP_SAFE_CALL(hipMemcpy(
            data_,
            rhs.data(),
            sizeof(T) * size_,
            hipMemcpyDeviceToDevice
            ));
        HIP_SAFE_CALL(hipDeviceSynchronize());
    }
    return *this;
}

template <typename T>
device_vector<T>& device_vector<T>::operator=(device_vector<T>&& rhs)
{
    if (&rhs != this)
    {
        data_ = std::move(rhs.data_);
        size_ = rhs.size_;

        rhs.data_ = nullptr;
        rhs.size_ = 0;
    }
    return *this;
}

template <typename T>
template <typename A>
device_vector<T>& device_vector<T>::operator=(std::vector<T, A> const& rhs)
{
    size_ = rhs.size();
    HIP_SAFE_CALL(hipMalloc(&data_, sizeof(T) * size_));
    HIP_SAFE_CALL(hipMemcpy(
        data_,
        rhs.data(),
        sizeof(T) * size_,
        hipMemcpyHostToDevice
        ));
    return *this;
}

template <typename T>
void device_vector<T>::resize(size_t size)
{
    if (size_ == size)
        return;

    T* prev{nullptr};
    size_t copy_size{0};
    if (size_ > 0)
    {
        copy_size = std::min(size_, size);
        HIP_SAFE_CALL(hipMalloc(&prev, copy_size * sizeof(T)));
        HIP_SAFE_CALL(hipMemcpy(
            prev,
            data_,
            copy_size * sizeof(T),
            hipMemcpyDeviceToDevice
            ));
        HIP_SAFE_CALL(hipDeviceSynchronize());
    }

    size_ = size;
    HIP_SAFE_CALL(hipFree(data_));
    HIP_SAFE_CALL(hipMalloc(&data_, sizeof(T) * size_));

    if (prev && copy_size > 0)
    {
        HIP_SAFE_CALL(hipMemcpy(
            data_,
            prev,
            copy_size * sizeof(T),
            hipMemcpyDeviceToDevice
            ));
        HIP_SAFE_CALL(hipDeviceSynchronize());
        HIP_SAFE_CALL(hipFree(prev));
    }
}

template <typename T>
void device_vector<T>::resize(size_t size, T const& value)
{
    size_t prev_size = size_;

    resize(size);

    if (prev_size < size_)
    {
        size_t more = size_ - prev_size;
        hip::fill(data_ + prev_size, more * sizeof(T), &value, sizeof(value));
    }
}

template <typename T>
T* device_vector<T>::data()
{
    return data_;
}

template <typename T>
T const* device_vector<T>::data() const
{
    return data_;
}

template <typename T>
size_t device_vector<T>::size() const
{
    return size_;
}

template <typename T>
bool device_vector<T>::empty() const
{
    return size_ == 0;
}

template <typename T>
T* device_vector<T>::begin()
{
    return data_;
}

template <typename T>
T* device_vector<T>::end()
{
    return data_ + size_;
}

template <typename T>
T const* device_vector<T>::begin() const
{
    return data_;
}

template <typename T>
T const* device_vector<T>::end() const
{
    return data_ + size_;
}

template <typename T>
T const* device_vector<T>::cbegin() const
{
    return data_;
}

template <typename T>
T const* device_vector<T>::cend() const
{
    return data_ + size_;
}

template <typename T>
VSNRAY_GPU_FUNC
T& device_vector<T>::operator[](size_t pos)
{
    return data_[pos];
}

template <typename T>
VSNRAY_GPU_FUNC
T const& device_vector<T>::operator[](size_t pos) const
{
    return data_[pos];
}

} // hip
} // visionaray
