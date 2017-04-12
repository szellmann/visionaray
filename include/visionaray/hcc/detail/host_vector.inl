// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{
namespace hcc
{

template <typename T, typename Alloc>
template <typename T2, typename Alloc2>
VSNRAY_CPU_FUNC
host_vector<T, Alloc>::host_vector(device_vector<T2, Alloc2> const& rhs)
    : host_vector(rhs.size())
{
    Alloc2 alloc2 = rhs.get_allocator();
    hc::accelerator_view av = alloc2.accelerator().get_default_view();
    av.copy(rhs.data(), base_type::data(), base_type::size() * sizeof(T));
}

template <typename T, typename Alloc>
template <typename T2, typename Alloc2>
VSNRAY_CPU_FUNC
host_vector<T, Alloc>& host_vector<T, Alloc>::operator=(device_vector<T2, Alloc2> const& rhs)
{
    base_type::resize(rhs.size());

    Alloc2 alloc2 = rhs.get_allocator();
    hc::accelerator_view av = alloc2.accelerator().get_default_view();
    av.copy(rhs.data(), base_type::data(), base_type::size() * sizeof(T));

    return *this;
}

} // hcc
} // visionaray
