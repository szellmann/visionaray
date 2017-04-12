// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iterator>
#include <utility>

#include <hcc/hc.hpp>

namespace visionaray
{
namespace hcc
{

//-------------------------------------------------------------------------------------------------
// device_vector members
//

template <typename T, typename Alloc>
VSNRAY_CPU_FUNC
device_vector<T, Alloc>::device_vector(typename device_vector<T, Alloc>::size_type n)
    : data_(alloc_.allocate(n))
    , size_(n)
    , capacity_(n)
{
}

template <typename T, typename Alloc>
VSNRAY_CPU_FUNC
device_vector<T, Alloc>::device_vector(
        typename device_vector<T, Alloc>::size_type n,
        typename device_vector<T, Alloc>::value_type const& value
        )
    : device_vector(n)
{
    // memset
    auto& data = data_;
    hc::parallel_for_each(
            hc::extent<1>(n),
            [=](hc::index<1> idx) [[hc]] { data[idx[0]] = value; }
            );
}

template <typename T, typename Alloc>
VSNRAY_CPU_FUNC
device_vector<T, Alloc>::device_vector(device_vector const& rhs)
    : device_vector(rhs.size())
{
    hc::accelerator_view av = alloc_.accelerator().get_default_view();
    av.copy(rhs.data(), data_, size_ * sizeof(T));
}

template <typename T, typename Alloc>
VSNRAY_CPU_FUNC
device_vector<T, Alloc>::device_vector(device_vector&& rhs)
    : alloc_(std::move(rhs.data_))
    , data_(std::move(rhs.data_))
    , size_(rhs.size_)
    , capacity_(rhs.capacity_)
{
}

template <typename T, typename Alloc>
template <typename T2, typename Alloc2>
VSNRAY_CPU_FUNC
device_vector<T, Alloc>::device_vector(std::vector<T2, Alloc2> const& rhs)
    : device_vector(rhs.size())
{
    hc::accelerator_view av = alloc_.accelerator().get_default_view();
    av.copy(rhs.data(), data_, size_ * sizeof(T));
}

template <typename T, typename Alloc>
template <typename It>
VSNRAY_CPU_FUNC
device_vector<T, Alloc>::device_vector(It first, It last)
    : device_vector(std::distance(first, last))
{
    // TODO
}

template <typename T, typename Alloc>
VSNRAY_CPU_FUNC
device_vector<T, Alloc>& device_vector<T, Alloc>::operator=(device_vector<T, Alloc> const& rhs)
{
    if (&rhs != this)
    {
        alloc_    = rhs.alloc_;
        data_     = alloc_.allocate(rhs.size_);
        size_     = rhs.size_;
        capacity_ = rhs.capacity_;

        hc::accelerator_view av = alloc_.accelerator().get_default_view();
        av.copy(rhs.data(), data_, size_ * sizeof(T));
    }
    return *this;
}

template <typename T, typename Alloc>
VSNRAY_CPU_FUNC
device_vector<T, Alloc>& device_vector<T, Alloc>::operator=(device_vector<T, Alloc>&& rhs)
{
    if (&rhs != this)
    {
        alloc_    = std::move(rhs.alloc_);
        data_     = std::move(rhs.data_);
        size_     = rhs.size_;
        capacity_ = rhs.capacity_;
    }
    return *this;
}

template <typename T, typename Alloc>
template <typename T2, typename Alloc2>
VSNRAY_CPU_FUNC
device_vector<T, Alloc>& device_vector<T, Alloc>::operator=(std::vector<T2, Alloc2> const& rhs)
{
    data_     = alloc_.allocate(rhs.size());
    size_     = rhs.size();
    capacity_ = size_;

    hc::accelerator_view av = alloc_.accelerator().get_default_view();
    av.copy(rhs.data(), data_, size_ * sizeof(T));

    return *this;
}

template <typename T, typename Alloc>
VSNRAY_CPU_FUNC
void device_vector<T, Alloc>::resize(
        typename device_vector<T, Alloc>::size_type n,
        typename device_vector<T, Alloc>::value_type x
        )
{
    reserve(n);

    // memset
    auto& data = data_;
    hc::parallel_for_each(
            hc::extent<1>(n),
            [=](hc::index<1> idx) [[hc]] { data[idx[0]] = x; }
            );

    size_ = n;
}

template <typename T, typename Alloc>
VSNRAY_FUNC
typename device_vector<T, Alloc>::size_type device_vector<T, Alloc>::size() const
{
    return size_;
}

template <typename T, typename Alloc>
VSNRAY_FUNC
typename device_vector<T, Alloc>::size_type device_vector<T, Alloc>::max_size() const
{
    return size_type(-1);
}

template <typename T, typename Alloc>
VSNRAY_CPU_FUNC
void device_vector<T, Alloc>::reserve(typename device_vector<T, Alloc>::size_type n)
{
    if (n > capacity_)
    {
        alloc_.deallocate(data_, capacity_);
        data_ = alloc_.allocate(n); // TODO
        capacity_ = n;
    }
}

template <typename T, typename Alloc>
VSNRAY_FUNC
typename device_vector<T, Alloc>::size_type device_vector<T, Alloc>::capacity() const
{
    return capacity_;
}

template <typename T, typename Alloc>
VSNRAY_FUNC
void device_vector<T, Alloc>::shrink_to_fit() const
{
    if (capacity_ > size_)
    {
        T* tmp = alloc_.allocate(size_);
        hc::accelerator_view av = alloc_.accelerator().get_default_view();
        av.copy(data_, tmp, size_ * sizeof(T));
        alloc_.deallocate(data_, capacity_);
        data_ = tmp;
    }
}

template <typename T, typename Alloc>
VSNRAY_FUNC
typename device_vector<T, Alloc>::reference device_vector<T, Alloc>::operator[](
        typename device_vector<T, Alloc>::size_type n
        )
{
    return data_[n];
}

template <typename T, typename Alloc>
VSNRAY_FUNC
typename device_vector<T, Alloc>::const_reference device_vector<T, Alloc>::operator[](
        typename device_vector<T, Alloc>::size_type n
        ) const
{
    return data_[n];
}

// TODO: iterator interface

template <typename T, typename Alloc>
VSNRAY_FUNC
typename device_vector<T, Alloc>::pointer device_vector<T, Alloc>::data()
{
    return data_;
}

template <typename T, typename Alloc>
VSNRAY_FUNC
typename device_vector<T, Alloc>::const_pointer device_vector<T, Alloc>::data() const
{
    return data_;
}

template <typename T, typename Alloc>
VSNRAY_CPU_FUNC
void device_vector<T, Alloc>::clear()
{
    size_ = 0;
}

template <typename T, typename Alloc>
VSNRAY_FUNC
bool device_vector<T, Alloc>::empty() const
{
    return size_ == 0;
}

template <typename T, typename Alloc>
VSNRAY_CPU_FUNC
void device_vector<T, Alloc>::push_back(typename device_vector<T, Alloc>::value_type const& x)
{
    reserve(size_ + 1);

    hc::accelerator_view av = alloc_.accelerator().get_default_view();
    av.copy(&x, data_ + size_, sizeof(T));
}

} // hcc
} // visionaray
