// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_TEXTURE_COMMON_H
#define VSNRAY_TEXTURE_COMMON_H

#include <cstddef>

#include <algorithm>
#include <vector>

#include <visionaray/detail/aligned_vector.h>

#include "../forward.h"


namespace visionaray
{

namespace detail
{

// TODO: lambda
template <typename S, typename T>
struct cast
{
    float operator()(T val) { return static_cast<S>(val); }
};

} // detail


class texture_params_base
{
public:

    texture_params_base()
        : address_mode_(Wrap)
        , filter_mode_(Nearest)
    {}

    void set_address_mode(tex_address_mode mode) { address_mode_ = mode; }
    tex_address_mode get_address_mode() const { return address_mode_; }

    void set_filter_mode(tex_filter_mode mode) { filter_mode_ = mode; }
    tex_filter_mode get_filter_mode() const { return filter_mode_; }

protected:

    tex_address_mode address_mode_;
    tex_filter_mode filter_mode_;

};


template <typename T>
class texture_base : public texture_params_base
{
public:

    using value_type = T;

    texture_base(size_t size)
        : data_(aligned_vector<T>(size))
    {
    }

    void set_data(value_type const* data) { std::copy( data, data + data_.size(), data_.begin() ); }
    value_type const* data() const { return data_.data(); }

protected:

    aligned_vector<T> data_;

};

template <typename T>
class texture_ref_base : public texture_params_base
{
public:

    typedef T value_type;


    texture_ref_base(size_t size = 0)
        : data_(nullptr)
    {
    }

    void set_data(value_type const* data) { data_ = data; } // TODO: initialize through c'tor
    value_type const* data() const { return data_; }

protected:

    value_type const* data_;

};


template <typename T, typename Derived>
class prefilterable
{
public:

    typedef T value_type;
    typedef short element_type;

    element_type* prefiltered_data;


    prefilterable() : prefiltered_data(0), filter_mode_(Nearest) {}

    void set_filter_mode(tex_filter_mode mode)
    {
        if (mode == BSplineInterpol)
        {
            Derived* d = static_cast<Derived*>(this);

            prefiltered_.resize( d->size() );
            prefiltered_data = &prefiltered_[0];
            std::transform( &(d->data)[0], &(d->data)[d->size()], prefiltered_.begin(), detail::cast<element_type, value_type>() );
            convert_for_bspline_interpol( d );
        }

        filter_mode_ = mode;
    }

    tex_filter_mode get_filter_mode() const { return filter_mode_; }

protected:

    tex_filter_mode filter_mode_;
    std::vector<element_type> prefiltered_;

};


} // visionaray


#endif // VSNRAY_TEXTURE_COMMON_H
