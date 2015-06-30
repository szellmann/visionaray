// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_TEXTURE_COMMON_H
#define VSNRAY_TEXTURE_COMMON_H

#include <cassert>
#include <cstddef>

#include <algorithm>
#include <array>
#include <vector>

#include <visionaray/aligned_vector.h>
#include <visionaray/pixel_format.h>

#include "../forward.h"
#include "swizzle.h"


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


template <size_t Dim>
class texture_params_base
{
public:

    texture_params_base()
        : filter_mode_(Nearest)
    {
        for (size_t d = 0; d < Dim; ++d)
        {
            address_mode_[d] = Wrap;
        }
    }

    void set_address_mode(size_t index, tex_address_mode mode)
    {
        assert( index < Dim );
        address_mode_[index] = mode;
    }

    void set_address_mode(tex_address_mode mode)
    {
        for (size_t d = 0; d < Dim; ++d)
        {
            address_mode_[d] = mode;
        }
    }

    tex_address_mode get_address_mode(size_t index) const
    {
        assert( index < Dim );
        return address_mode_[index];
    }

    std::array<tex_address_mode, Dim> const& get_address_mode() const
    {
        return address_mode_;
    }

    void set_filter_mode(tex_filter_mode mode)
    {
        filter_mode_ = mode;
    }

    tex_filter_mode get_filter_mode() const
    {
        return filter_mode_;
    }

protected:

    std::array<tex_address_mode, Dim>   address_mode_;
    tex_filter_mode                     filter_mode_;

};


template <typename T, size_t Dim>
class texture_base : public texture_params_base<Dim>
{
public:

    using value_type = T;

    texture_base(size_t size)
        : data_(aligned_vector<T>(size))
    {
    }

    void set_data(T const* data)
    {
        std::copy( data, data + data_.size(), data_.begin() );
    }

    void set_data(T const* data, pixel_format format, pixel_format internal_format)
    {
        if (format != internal_format)
        {
            // Swizzle in-place
            aligned_vector<T> tmp( data, data + data_.size() );
            swizzle( tmp.data(), tmp.size(), format, internal_format );
            set_data( tmp.data() );
        }
        else
        {
            // Simple copy
            set_data( data );
        }
    }

    template <typename U>
    void set_data(U const* data, pixel_format format, pixel_format internal_format)
    {
        // Copy to temporary array, then swizzle
        aligned_vector<T> dst( data_.size() );
        swizzle( dst.data(), data, dst.size(), format, internal_format );
        set_data( dst.data() );
    }

    value_type const* data() const
    {
        return data_.data();
    }

protected:

    aligned_vector<T> data_;

};

template <typename T, size_t Dim>
class texture_ref_base : public texture_params_base<Dim>
{
public:

    using value_type = T;
    using base_type  = texture_params_base<Dim>;

public:

    texture_ref_base(texture_base<T, Dim> const& tex)
        : base_type(tex)
        , data_(tex.data())
    {
    }

    texture_ref_base(size_t size = 0)
        : data_(nullptr)
    {
        VSNRAY_UNUSED(size);
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
