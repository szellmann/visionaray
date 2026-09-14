// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <utility>

#include <hip/hip_runtime.h>

#include "../fill.h"
#include "../safe_call.h"

namespace visionaray
{
namespace hip
{

template <typename T, typename Alloc>
device_vector<T, Alloc>::~device_vector()
{
    HIP_SAFE_CALL(hipFree(data_));
}

template <typename T, typename Alloc>
device_vector<T, Alloc>::device_vector(device_vector<T, Alloc> const& rhs)
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

template <typename T, typename Alloc>
device_vector<T, Alloc>::device_vector(device_vector<T, Alloc>&& rhs)
{
    if (&rhs != this)
    {
        data_ = std::move(rhs.data_);
        size_ = rhs.size_;

        rhs.data_ = nullptr;
        rhs.size_ = 0;
    }
}

template <typename T, typename Alloc>
device_vector<T, Alloc>::device_vector(size_t size)
    : size_(size)
{
    HIP_SAFE_CALL(hipMalloc(&data_, sizeof(T) * size_));
}

template <typename T, typename Alloc>
device_vector<T, Alloc>::device_vector(size_t size, T const& value)
    : size_(size)
{
    HIP_SAFE_CALL(hipMalloc(&data_, sizeof(T) * size_));
    hip::fill(data_, size_ * sizeof(T), (T*)&value, sizeof(value));
}

template <typename T, typename Alloc>
template <typename A>
device_vector<T, Alloc>::device_vector(std::vector<T, A> const &hv)
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

template <typename T, typename Alloc>
template <typename A>
device_vector<T, Alloc>::operator std::vector<T, A>() const
{
    std::vector<T, A> hv(size_);
    HIP_SAFE_CALL(hipMemcpy(
        hv.data(),
        data_,
        sizeof(T) * size_,
        hipMemcpyDeviceToHost
        ));
    return hv;
}

template <typename T, typename Alloc>
device_vector<T, Alloc>::device_vector(const T* data, size_t size)
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

template <typename T, typename Alloc>
device_vector<T, Alloc>::device_vector(const T* begin, const T* end)
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

template <typename T, typename Alloc>
device_vector<T, Alloc>& device_vector<T, Alloc>::operator=(device_vector<T, Alloc> const& rhs)
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

template <typename T, typename Alloc>
device_vector<T, Alloc>& device_vector<T, Alloc>::operator=(device_vector<T, Alloc>&& rhs)
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

template <typename T, typename Alloc>
template <typename A>
device_vector<T, Alloc>& device_vector<T, Alloc>::operator=(std::vector<T, A> const& rhs)
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

template <typename T, typename Alloc>
void device_vector<T, Alloc>::reserve(size_t size)
{
    if (size <= capacity_)
    {
        return;
    }

    T* prev{nullptr};
    size_t copy_size{0};
    if (capacity_ > 0)
    {
        copy_size = std::min(capacity_, size);
        HIP_SAFE_CALL(hipMalloc(&prev, copy_size * sizeof(T)));
        HIP_SAFE_CALL(hipMemcpy(
            prev,
            data_,
            copy_size * sizeof(T),
            hipMemcpyDeviceToDevice
            ));
        HIP_SAFE_CALL(hipDeviceSynchronize());
    }

    capacity_ = size;
    HIP_SAFE_CALL(hipFree(data_));
    HIP_SAFE_CALL(hipMalloc(&data_, sizeof(T) * capacity_));

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

template <typename T, typename Alloc>
void device_vector<T, Alloc>::resize(size_t size)
{
    if (size_ == size)
        return;

    reserve(size);
    size_ = size;
}

template <typename T, typename Alloc>
void device_vector<T, Alloc>::resize(size_t size, T const& value)
{
    size_t prev_size = size_;

    resize(size);

    if (prev_size < size_)
    {
        size_t more = size_ - prev_size;
        hip::fill(data_ + prev_size, more * sizeof(T), &value, sizeof(value));
    }
}

template <typename T, typename Alloc>
void device_vector<T, Alloc>::push_back(T const& value)
{
  resize(size_ + 1);

  HIP_SAFE_CALL(hipMemcpy(
        data_ + size_,
        &value,
        sizeof(T),
        hipMemcpyHostToDevice
        ));
}

template <typename T, typename Alloc>
template <typename... Args>
void device_vector<T, Alloc>::emplace_back(Args&&... args)
{
  T value(std::forward<Args>(args)...);
  resize(size_ + 1);

  HIP_SAFE_CALL(hipMemcpy(
        data_ + size_,
        &value,
        sizeof(T),
        hipMemcpyHostToDevice
        ));
}

template <typename T, typename Alloc>
void device_vector<T, Alloc>::clear()
{
    HIP_SAFE_CALL(hipFree(data_));
    capacity_ = 0;
    size_ = 0;
}

template <typename T, typename Alloc>
T* device_vector<T, Alloc>::data()
{
    return data_;
}

template <typename T, typename Alloc>
T const* device_vector<T, Alloc>::data() const
{
    return data_;
}

template <typename T, typename Alloc>
size_t device_vector<T, Alloc>::size() const
{
    return size_;
}

template <typename T, typename Alloc>
bool device_vector<T, Alloc>::empty() const
{
    return size_ == 0;
}

template <typename T, typename Alloc>
T* device_vector<T, Alloc>::begin()
{
    return data_;
}

template <typename T, typename Alloc>
T* device_vector<T, Alloc>::end()
{
    return data_ + size_;
}

template <typename T, typename Alloc>
T const* device_vector<T, Alloc>::begin() const
{
    return data_;
}

template <typename T, typename Alloc>
T const* device_vector<T, Alloc>::end() const
{
    return data_ + size_;
}

template <typename T, typename Alloc>
T const* device_vector<T, Alloc>::cbegin() const
{
    return data_;
}

template <typename T, typename Alloc>
T const* device_vector<T, Alloc>::cend() const
{
    return data_ + size_;
}

template <typename T, typename Alloc>
VSNRAY_GPU_FUNC
T& device_vector<T, Alloc>::operator[](size_t pos)
{
    return data_[pos];
}

template <typename T, typename Alloc>
VSNRAY_GPU_FUNC
T const& device_vector<T, Alloc>::operator[](size_t pos) const
{
    return data_[pos];
}

} // hip
} // visionaray
