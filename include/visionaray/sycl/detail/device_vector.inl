// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <sycl/sycl.hpp>

//#include "../fill.h"

namespace visionaray
{
namespace sycl
{

template <typename T>
device_vector<T>::~device_vector()
{
    ::sycl::free(data_, queue_);
}

template <typename T>
device_vector<T>::device_vector(device_vector<T> const& rhs)
    : size_(rhs.size())
{
    if (&rhs != this)
    {
        data_ = ::sycl::malloc_device<T>(size_, queue_);
        queue_.memcpy(data_, rhs.data_, sizeof(T) * size_).wait();
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
    data_ = ::sycl::malloc_device<T>(size_, queue_);
}

template <typename T>
device_vector<T>::device_vector(size_t size, T const& value)
    : size_(size)
{
    data_ = ::sycl::malloc_device<T>(size_, queue_);
    //cuda::fill(data_, size_ * sizeof(T), (T*)&value, sizeof(value));
}

template <typename T>
template <typename A>
device_vector<T>::device_vector(std::vector<T, A> const &hv)
    : size_(hv.size())
{
    data_ = ::sycl::malloc_device<T>(size_, queue_);
    queue_.memcpy(data_, hv.data(), sizeof(T) * size_).wait();
}

template <typename T>
device_vector<T>::device_vector(const T* data, size_t size)
    : size_(size)
{
    data_ = ::sycl::malloc_device<T>(size_, queue_);
    queue_.memcpy(data_, data, sizeof(T) * size_).wait();
}

template <typename T>
device_vector<T>::device_vector(const T* begin, const T* end)
    : size_(end - begin)
{
    data_ = ::sycl::malloc_device<T>(size_, queue_);
    queue_.memcpy(data_, begin, sizeof(T) * size_).wait();
}

template <typename T>
device_vector<T>& device_vector<T>::operator=(device_vector<T> const& rhs)
{
    if (&rhs != this)
    {
        size_ = rhs.size_;
        data_ = ::sycl::malloc_device<T>(size_, queue_);
        queue_.memcpy(data_, rhs.data(), sizeof(T) * size_).wait();
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
void device_vector<T>::resize(size_t size)
{
    if (size_ == size)
        return;

    T* prev{nullptr};
    size_t copy_size{0};
    if (size_ > 0)
    {
        copy_size = std::min(size_, size);
        prev = ::sycl::malloc_device<T>(copy_size, queue_);
        queue_.memcpy(prev, data_, copy_size * sizeof(T)).wait();
    }

    size_ = size;
    ::sycl::free(data_, queue_);
    data_ = ::sycl::malloc_device<T>(size_, queue_);

    if (prev && copy_size > 0)
    {
        queue_.memcpy(data_, prev, copy_size * sizeof(T)).wait();
        ::sycl::free(prev, queue_);
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
        //cuda::fill(data_ + prev_size, more * sizeof(T), &value, sizeof(value));
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

} // sycl
} // visionaray
