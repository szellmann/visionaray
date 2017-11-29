// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VARIANT_H
#define VSNRAY_VARIANT_H 1

#include <type_traits>

#include "detail/macros.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// A most simple variant type that is compatible with CUDA
// Only suitable for PODs!
// Not recursive!
//

namespace detail
{

//-------------------------------------------------------------------------------------------------
// Get index of T in parameter pack
//

template <typename ...Ts>
struct index_of
{
    enum { value = 0 };
};

template <typename U, typename ...Ts>
struct index_of<U, U, Ts...>
{
    enum { value = 1 };
};

template <typename U, typename T, typename ...Ts>
struct index_of<U, T, Ts...>
{
    enum { v = index_of<U, Ts...>::value };
    enum { value = v == 0 ? 0 : 1 + v };
};


//-------------------------------------------------------------------------------------------------
// Get type at index I
//

template <unsigned I, typename ...Ts>
struct type_at_impl
{
};

template <typename T, typename ...Ts>
struct type_at_impl<1, T, Ts...>
{
    using type = T;
};

template <unsigned I, typename T, typename ...Ts>
struct type_at_impl<I, T, Ts...> : type_at_impl<I - 1, Ts...>
{
};

template <unsigned I, typename ...Ts>
using type_at = typename type_at_impl<I, Ts...>::type;


//-------------------------------------------------------------------------------------------------
//
//

template <unsigned I>
using type_index = std::integral_constant<unsigned, I>;



//-------------------------------------------------------------------------------------------------
// Recursive union storage
//

template <typename ...Ts>
union variant_storage
{
};

template <typename T, typename ...Ts>
union variant_storage<T, Ts...>
{
    T element;
    variant_storage<Ts...> elementN;

    // access

    VSNRAY_FUNC T& get(type_index<1>)
    {
        return element;
    }

    VSNRAY_FUNC T const& get(type_index<1>) const
    {
        return element;
    }

    template <unsigned I>
    VSNRAY_FUNC type_at<I - 1, Ts...>& get(type_index<I>)
    {
        return elementN.get(type_index<I - 1>{});
    }

    template <unsigned I>
    VSNRAY_FUNC type_at<I - 1, Ts...> const& get(type_index<I>) const
    {
        return elementN.get(type_index<I - 1>{});
    }
};

} // detail


template <typename ...Ts>
class variant
{
public:

    variant() = default;

    template <typename T>
    VSNRAY_FUNC variant(T const& value)
        : type_index_(detail::index_of<T, Ts...>::value)
    {
        storage_.get(detail::type_index<detail::index_of<T, Ts...>::value>()) = value;
    }

    template <typename T>
    VSNRAY_FUNC variant& operator=(T const& value)
    {
        type_index_ = detail::index_of<T, Ts...>::value;
        storage_.get(detail::type_index<detail::index_of<T, Ts...>::value>()) = value;
        return *this;
    }

    template <typename T>
    VSNRAY_FUNC T* as()
    {
        return type_index_ == detail::index_of<T, Ts...>::value
            ? &storage_.get(detail::type_index<detail::index_of<T, Ts...>::value>())
            : nullptr;
    }

    template <typename T>
    VSNRAY_FUNC T const* as() const
    {
        return type_index_ == detail::index_of<T, Ts...>::value
            ? &storage_.get(detail::type_index<detail::index_of<T, Ts...>::value>())
            : nullptr;
    }

private:

    detail::variant_storage<Ts...>  storage_;
    unsigned                        type_index_;

};


template <unsigned I, typename ...Ts>
struct apply_visitor_impl;

template <unsigned I, typename T, typename ...Ts>
struct apply_visitor_impl<I, T, Ts...>
{
    template <typename Visitor, typename Variant>
    VSNRAY_FUNC
    typename Visitor::return_type operator()(Visitor const& visitor, Variant const& var) const
    {
        auto ptr = var.template as<T>();

        if (ptr)
        {
            return visitor(*ptr);
        }
        else
        {
            return apply_visitor_impl<I - 1, Ts...>()(visitor, var);
        }
    }
};

template <>
struct apply_visitor_impl<0>
{
    template <typename Visitor, typename Variant>
    VSNRAY_FUNC
    typename Visitor::return_type operator()(Visitor const&, Variant const&)
    {
        // TODO
        return typename Visitor::return_type();
    }
};

template <typename Visitor, typename ...Ts>
VSNRAY_FUNC
typename Visitor::return_type apply_visitor(Visitor const& visitor, variant<Ts...> const& var)
{
    return apply_visitor_impl<sizeof...(Ts), Ts...>()(visitor, var);
}

} // visionaray

#endif // VSNRAY_VARIANT_H
