// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HCC_REVERSE_ITERATOR_H
#define VSNRAY_HCC_REVERSE_ITERATOR_H 1

#include <iterator>

#include "../detail/macros.h"

namespace visionaray
{
namespace hcc
{

//-------------------------------------------------------------------------------------------------
// Reverse the direction of a given iterator
//

template <typename It>
class reverse_iterator
{
public:

    typedef typename std::iterator_traits<It>::value_type           value_type;
    typedef typename std::iterator_traits<It>::difference_type      difference_type;
    typedef typename std::iterator_traits<It>::pointer              pointer;
    typedef typename std::iterator_traits<It>::reference            reference;
    typedef typename std::iterator_traits<It>::iterator_category    iterator_category;
    typedef It                                                      iterator_type;

public:

    // Default constructor.
    VSNRAY_FUNC
    reverse_iterator() = default;

    // Construct from base iterator.
    VSNRAY_FUNC
    explicit reverse_iterator(It x);

    // Copy constructor.
    template <typename OtherIt>
    VSNRAY_FUNC
    reverse_iterator(reverse_iterator<OtherIt> const& rhs);

    // Assignment.
    template <typename OtherIt>
    VSNRAY_FUNC
    reverse_iterator& operator=(reverse_iterator<OtherIt> const& rhs);

    // Access underlying base iterator.
    VSNRAY_FUNC
    It base() const;

    // Access the pointed-to element.
    VSNRAY_FUNC
    reference operator*() const;

    // Access the pointed-to element.
    VSNRAY_FUNC
    pointer operator->() const;

    // Access an element by index.
    VSNRAY_FUNC
    value_type operator[](difference_type n) const;

    // Increments the iterator (pre).
    VSNRAY_FUNC
    reverse_iterator& operator++();

    // Decrements the iterator (pre).
    VSNRAY_FUNC
    reverse_iterator& operator--();

    // Increments the iterator (post).
    VSNRAY_FUNC
    reverse_iterator operator++(int);

    // Decrements the iterator (post).
    VSNRAY_FUNC
    reverse_iterator operator--(int);

    // Advances the iterator by n positions.
    VSNRAY_FUNC
    reverse_iterator operator+(difference_type n) const;

    // Advances the iterator by -n positions.
    VSNRAY_FUNC
    reverse_iterator operator-(difference_type n) const;

    // Advance by n and assignment.
    VSNRAY_FUNC
    reverse_iterator& operator+=(difference_type n) const;

    // Advance by -n and assignment.
    VSNRAY_FUNC
    reverse_iterator& operator-=(difference_type n) const;

protected:

    It current;

};

} // hcc
} // visionaray

#include "detail/reverse_iterator.inl"

#endif // VSNRAY_HCC_REVERSE_ITERATOR_H
