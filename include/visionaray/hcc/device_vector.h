// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HCC_DEVICE_VECTOR_H
#define VSNRAY_HCC_DEVICE_VECTOR_H 1

#include <vector>

#include "device_malloc_allocator.h"
#include "reverse_iterator.h"

namespace visionaray
{
namespace hcc
{

//-------------------------------------------------------------------------------------------------
// Mini-me of thrust::device_vector
//

template<typename T, typename Alloc = hcc::device_malloc_allocator<T>>
class device_vector
{
public:

    typedef typename Alloc::value_type              value_type;
    typedef typename Alloc::pointer                 pointer;
    typedef typename Alloc::const_pointer           const_pointer;
    typedef typename Alloc::reference               reference;
    typedef typename Alloc::const_reference         const_reference;
    typedef typename Alloc::size_type               size_type;
    typedef typename Alloc::difference_type         difference_type;
    typedef Alloc                                   allocator_type;

    typedef pointer                                 iterator;
    typedef const_pointer                           const_iterator;

    typedef hcc::reverse_iterator<iterator>         reverse_iterator;
    typedef hcc::reverse_iterator<const_iterator>   const_reverse_iterator;

public:

    // Default constructor.
    VSNRAY_CPU_FUNC
    device_vector() = default;

    // Constructor that allocate n elements.
    VSNRAY_CPU_FUNC
    explicit device_vector(size_type n);

    // Construct with n copies of an exemplar element.
    VSNRAY_CPU_FUNC
    explicit device_vector(size_type n, value_type const& value);

    // Copy constructor.
    VSNRAY_CPU_FUNC
    device_vector(device_vector const& rhs);

    // Move constructor.
    VSNRAY_CPU_FUNC
    device_vector(device_vector&& rhs);

    // TODO: device-ctor
    //

    // Destructor, erases all elements and deallocates capacity.
    VSNRAY_CPU_FUNC
    ~device_vector();

    // Copy-constuct from an std::vector.
    template <typename T2, typename Alloc2>
    VSNRAY_CPU_FUNC
    device_vector(std::vector<T2, Alloc2> const& rhs);

    // Construct from an iterator range.
    template <typename It>
    VSNRAY_FUNC
    device_vector(It first, It last);


    // Assignment.
    VSNRAY_CPU_FUNC
    device_vector& operator=(device_vector const& rhs);

    // Move assignment.
    VSNRAY_CPU_FUNC
    device_vector& operator=(device_vector&& rhs);

    // TODO: device-assignment
    //

    // Assign an std::vector.
    template <typename T2, typename Alloc2>
    VSNRAY_CPU_FUNC
    device_vector& operator=(std::vector<T2, Alloc2> const& rhs);

    // Resize and fill with x.
    VSNRAY_CPU_FUNC
    void resize(size_type n, value_type const& x = value_type());

    // Returns the number of elements.
    VSNRAY_FUNC
    size_type size() const;

    // Returns the size of the largest possible vector.
    VSNRAY_FUNC
    size_type max_size() const;

    // Reserve storage for more elements (no effect if LE capacity()).
    VSNRAY_CPU_FUNC
    void reserve(size_type n);

    // Returns number of reseversed elements.
    VSNRAY_FUNC
    size_type capacity() const;

    // Shrink the capacity to exactly fit its elements.
    VSNRAY_CPU_FUNC
    void shrink_to_fit();

    // Access element at n.
    VSNRAY_FUNC
    reference operator[](size_type n);

    // Access element at n.
    VSNRAY_FUNC
    const_reference operator[](size_type n) const;

    // Returns an iterator to the first element of the vector.
    VSNRAY_FUNC
    iterator begin();

    // Returns an iterator to the first element of the vector.
    VSNRAY_FUNC
    const_iterator begin() const;

    // Returns an iterator to the first element of the vector.
    VSNRAY_FUNC
    const_iterator cbegin() const;

    // Returns a reverse iterator to the first element of the vector.
    VSNRAY_FUNC
    reverse_iterator rbegin();

    // Returns a reverse iterator to the first element of the vector.
    VSNRAY_FUNC
    const_reverse_iterator rbegin() const;

    // Returns a reverse iterator to the first element of the vector.
    VSNRAY_FUNC
    const_reverse_iterator crbegin() const;

    // Returns an iterator to the last+1 element of the vector.
    VSNRAY_FUNC
    iterator end();

    // Returns an iterator to the last+1 element of the vector.
    VSNRAY_FUNC
    const_iterator end() const;

    // Returns an iterator to the last+1 element of the vector.
    VSNRAY_FUNC
    const_iterator cend() const;

    // Returns a reverse iterator to the last+1 element of the vector.
    VSNRAY_FUNC
    iterator rend();

    // Returns a reverse iterator to the last+1 element of the vector.
    VSNRAY_FUNC
    const_iterator rend() const;

    // Returns a reverse iterator to the last+1 element of the vector.
    VSNRAY_FUNC
    const_iterator crend() const;

    // Returns a reference to the first element.
    VSNRAY_FUNC
    reference front();

    // Returns a reference to the first element.
    VSNRAY_FUNC
    const_reference front() const;

    // Returns a reference to the last element.
    VSNRAY_FUNC
    reference back();

    // Returns a reference to the last element.
    VSNRAY_FUNC
    const_reference back() const;

    // Returns a pointer to the underlying data.
    VSNRAY_FUNC
    pointer data();

    // Returns a pointer to the underlying data.
    VSNRAY_FUNC
    const_pointer data() const;

    // Clear the vector.
    VSNRAY_CPU_FUNC
    void clear();

    // Check if vector is empty.
    VSNRAY_CPU_FUNC
    bool empty() const;

    // Append an item to the end of the vector.
    VSNRAY_CPU_FUNC
    void push_back(value_type const& x);

    // Erase the last element in the vector (invalidates all iterators and references!)
    VSNRAY_CPU_FUNC
    void pop_back();

    // Swap two vectors.
    VSNRAY_CPU_FUNC
    void swap(device_vector& rhs);

    // Remove the element at position pos.
    VSNRAY_CPU_FUNC
    iterator erase(iterator pos);

    // Remove elements in range [first..last).
    VSNRAY_CPU_FUNC
    iterator erase(iterator first, iterator last);

    // TODO!
    //

    // Get a copy of the allocator
    VSNRAY_CPU_FUNC
    allocator_type get_allocator() const;

private:

    Alloc alloc_;
    T* data_ = 0;
    size_type size_ = 0;
    size_type capacity_ = 0;

};

} // hcc
} // visionaray

#include "detail/device_vector.inl"

#endif // VSNRAY_HCC_DEVICE_VECTOR_H
