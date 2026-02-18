// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cuda_runtime.h>

#include "../fill.h"
#include "../safe_call.h"

namespace visionaray
{
namespace cuda
{

template <typename T>
device_vector<T>::~device_vector()
{
    CUDA_SAFE_CALL(cudaFree(data_));
}

template <typename T>
device_vector<T>::device_vector(device_vector<T> const& rhs)
    : size_(rhs.size())
{
    if (&rhs != this)
    {
        CUDA_SAFE_CALL(cudaMalloc(&data_, sizeof(T) * size_));
        CUDA_SAFE_CALL(cudaMemcpy(
            data_,
            rhs.data(),
            sizeof(T) * size_,
            cudaMemcpyDeviceToDevice
            ));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
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
    CUDA_SAFE_CALL(cudaMalloc(&data_, sizeof(T) * size_));
}

template <typename T>
device_vector<T>::device_vector(size_t size, T const& value)
    : size_(size)
{
    CUDA_SAFE_CALL(cudaMalloc(&data_, sizeof(T) * size_));
    cuda::fill(data_, size_ * sizeof(T), (T*)&value, sizeof(value));
}

template <typename T>
template <typename A>
device_vector<T>::device_vector(std::vector<T, A> const &hv)
    : size_(hv.size())
{
    CUDA_SAFE_CALL(cudaMalloc(&data_, sizeof(T) * size_));
    CUDA_SAFE_CALL(cudaMemcpy(
        data_,
        hv.data(),
        sizeof(T) * size_,
        cudaMemcpyHostToDevice
        ));
}

template <typename T>
device_vector<T>::device_vector(const T* data, size_t size)
    : size_(size)
{
    CUDA_SAFE_CALL(cudaMalloc(&data_, sizeof(T) * size_));
    CUDA_SAFE_CALL(cudaMemcpy(
        data_,
        data,
        sizeof(T) * size_,
        cudaMemcpyDefault
        ));
}

template <typename T>
device_vector<T>::device_vector(const T* begin, const T* end)
    : size_(end - begin)
{
    CUDA_SAFE_CALL(cudaMalloc(&data_, sizeof(T) * size_));
    CUDA_SAFE_CALL(cudaMemcpy(
        data_,
        begin,
        sizeof(T) * size_,
        cudaMemcpyDefault
        ));

  cudaPointerAttributes attributes;
  CUDA_SAFE_CALL(cudaPointerGetAttributes(&attributes, begin));

  if (attributes.type == cudaMemoryTypeDevice)
  {
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
}

template <typename T>
device_vector<T>& device_vector<T>::operator=(device_vector<T> const& rhs)
{
    if (&rhs != this)
    {
        size_ = rhs.size_;
        CUDA_SAFE_CALL(cudaMalloc(&data_, sizeof(T) * size_));
        CUDA_SAFE_CALL(cudaMemcpy(
            data_,
            rhs.data(),
            sizeof(T) * size_,
            cudaMemcpyDeviceToDevice
            ));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
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
    CUDA_SAFE_CALL(cudaMalloc(&data_, sizeof(T) * size_));
    CUDA_SAFE_CALL(cudaMemcpy(
        data_,
        rhs.data(),
        sizeof(T) * size_,
        cudaMemcpyHostToDevice
        ));
    return *this;
}

template <typename T>
void device_vector<T>::reserve(size_t size)
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
        CUDA_SAFE_CALL(cudaMalloc(&prev, copy_size * sizeof(T)));
        CUDA_SAFE_CALL(cudaMemcpy(
            prev,
            data_,
            copy_size * sizeof(T),
            cudaMemcpyDeviceToDevice
            ));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    capacity_ = size;
    CUDA_SAFE_CALL(cudaFree(data_));
    CUDA_SAFE_CALL(cudaMalloc(&data_, sizeof(T) * capacity_));

    if (prev && copy_size > 0)
    {
        CUDA_SAFE_CALL(cudaMemcpy(
            data_,
            prev,
            copy_size * sizeof(T),
            cudaMemcpyDeviceToDevice
            ));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaFree(prev));
    }
}

template <typename T>
void device_vector<T>::resize(size_t size)
{
    if (size_ == size)
        return;

    reserve(size);
    size_ = size;
}

template <typename T>
void device_vector<T>::resize(size_t size, T const& value)
{
    size_t prev_size = size_;

    resize(size);

    if (prev_size < size_)
    {
        size_t more = size_ - prev_size;
        cuda::fill(data_ + prev_size, more * sizeof(T), &value, sizeof(value));
    }
}

template <typename T>
void device_vector<T>::push_back(T const& value)
{
  resize(size_ + 1);

  CUDA_SAFE_CALL(cudaMemcpy(
        data_ + size_,
        &value,
        sizeof(T),
        cudaMemcpyHostToDevice
        ));
}

template <typename T>
template <typename... Args>
void device_vector<T>::emplace_back(Args&&... args)
{
  T value(std::forward<Args>(args)...);
  resize(size_ + 1);

  CUDA_SAFE_CALL(cudaMemcpy(
        data_ + size_,
        &value,
        sizeof(T),
        cudaMemcpyHostToDevice
        ));
}

template <typename T>
void device_vector<T>::clear()
{
    CUDA_SAFE_CALL(cudaFree(data_));
    size_ = 0;
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

} // cuda
} // visionaray
