// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CUDA_DEVICE_VECTOR_H
#define VSNRAY_CUDA_DEVICE_VECTOR_H 1

#include <type_traits>
#include <utility>
#include <vector>

#include "../detail/macros.h"

namespace visionaray
{
namespace cuda
{

template <typename T>
class device_vector
{
    static_assert(std::is_trivially_copyable<T>::value);
public:
    typedef T value_type;
    typedef T* iterator;
    typedef T const* const_iterator;
    typedef T& reference;
    typedef T const& const_reference;
    typedef T* pointer;
    typedef T const* const_pointer;

    device_vector() = default;
   ~device_vector();
    device_vector(device_vector const& rhs);
    device_vector(device_vector&& rhs);

    device_vector(size_t size);
    device_vector(size_t size, T const& value);
    template <typename A>
    device_vector(std::vector<T, A> const& hv);
    device_vector(const T* data, size_t size);
    device_vector(const T* begin, const T* end);

    device_vector& operator=(device_vector const& rhs);
    device_vector& operator=(device_vector&& rhs);
    template <typename A>
    device_vector& operator=(std::vector<T, A> const& hv);

    void reserve(size_t size);
    void resize(size_t size);
    void resize(size_t size, T const& value);

    void push_back(T const& value);

    template<typename... Args>
    void emplace_back(Args&&... args);

    void clear();

    T* data();
    T const* data() const;

    size_t size() const;
    bool empty() const;

    T* begin();
    T* end();

    T const* begin() const;
    T const* end() const;

    T const* cbegin() const;
    T const* cend() const;

    VSNRAY_GPU_FUNC T& operator[](size_t pos);
    VSNRAY_GPU_FUNC T const& operator[](size_t pos) const;

private:
    T* data_{nullptr};
    size_t size_{0};
    size_t capacity_{0};
};

} // cuda
} // visionaray

#include "detail/device_vector.inl"

#endif // VSNRAY_CUDA_DEVICE_VECTOR_H
