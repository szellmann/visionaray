// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HCC_HOST_VECTOR_H
#define VSNRAY_HCC_HOST_VECTOR_H 1

#include <memory>
#include <vector>

#include "device_vector.h"

namespace visionaray
{
namespace hcc
{

//-------------------------------------------------------------------------------------------------
// Mini-me of thrust::host_vector
// For simplicity's sake implemented by publicly inheriting from std::vector
//

template <typename T, typename Alloc = std::allocator<T>>
class host_vector : public std::vector<T, Alloc>
{
private:

    typedef std::vector<T, Alloc> base_type;

public:

    // Inherit std::vector constructors.
    using base_type::base_type;

    // Default constructor.
    VSNRAY_CPU_FUNC
    host_vector() = default;

    // Copy-construct from an hcc::device_vector.
    template <typename T2, typename Alloc2>
    VSNRAY_CPU_FUNC
    host_vector(device_vector<T2, Alloc2> const& rhs);

    // Assign an hcc::device_vector.
    template <typename T2, typename Alloc2>
    host_vector& operator=(device_vector<T2, Alloc2> const& rhs);

};

} // hcc
} // visionaray

#include "detail/host_vector.inl"

#endif // VSNRAY_HCC_HOST_VECTOR_H
