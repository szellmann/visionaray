// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iterator>
#include <type_traits>

#include <visionaray/get_normal.h>
#include <visionaray/get_shading_normal.h>
#include <visionaray/prim_traits.h>
#include <visionaray/variant.h>

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Call get_normal for the underlying primitive
//

// w/o normals list (v1)
template <typename ReturnType, typename HR>
class get_normal_from_generic_primitive_visitor_v1
{
public:

    using return_type = ReturnType;

public:

    VSNRAY_FUNC
    get_normal_from_generic_primitive_visitor_v1(HR const&   hr)
        : hr_(hr)
    {
    }

    template <typename Primitive>
    VSNRAY_FUNC
    return_type operator()(Primitive const& primitive) const
    {
        return get_normal(hr_, primitive);
    }

private:

    HR const& hr_;

};


// w/ normals list (v2)
template <typename ReturnType, typename Normals, typename HR>
class get_normal_from_generic_primitive_visitor_v2
{
public:

    using return_type = ReturnType;

public:

    VSNRAY_FUNC
    get_normal_from_generic_primitive_visitor_v2(
            Normals     normals,
            HR const&   hr
            )
        : normals_(normals)
        , hr_(hr)
    {
    }

    template <typename Primitive>
    VSNRAY_FUNC
    return_type operator()(Primitive const& primitive) const
    {
        return get_normal(normals_, hr_, primitive);
    }

private:

    Normals   normals_;
    HR const& hr_;

};


//-------------------------------------------------------------------------------------------------
// Call get_shading_normal for the underlying primitive
//

template <typename ReturnType, typename NormalBinding, typename Normals, typename HR>
class get_shading_normal_from_generic_primitive_visitor
{
public:

    using return_type = ReturnType;

public:

    VSNRAY_FUNC
    get_shading_normal_from_generic_primitive_visitor(
            Normals     normals,
            HR const&   hr
            )
        : normals_(normals)
        , hr_(hr)
    {
    }

    template <typename Primitive>
    VSNRAY_FUNC
    return_type operator()(
            Primitive const& primitive
            ) const
    {
        return get_shading_normal(normals_, hr_, primitive, NormalBinding{});
    }

private:

    Normals   normals_;
    HR const& hr_;

};

} // detail


//-------------------------------------------------------------------------------------------------
// get_normal overloads
//

template <
    typename HR,
    typename ...Ts,
    typename P = detail::type_at<1, Ts...>
    >
VSNRAY_FUNC
inline auto get_normal(
        HR const&                   hr,
        generic_primitive<Ts...>    prim
        )
    -> decltype(get_normal(hr, P{}))
{
    using N = decltype(get_normal(hr, P{}));

    detail::get_normal_from_generic_primitive_visitor_v1<N, HR> visitor(hr);

    return apply_visitor(visitor, prim);
}

template <
    typename Normals,
    typename HR,
    typename ...Ts
    >
VSNRAY_FUNC
inline auto get_normal(
        Normals                     normals,
        HR const&                   hr,
        generic_primitive<Ts...>    prim
        )
    -> typename std::iterator_traits<Normals>::value_type
{
    detail::get_normal_from_generic_primitive_visitor_v2<
        typename std::iterator_traits<Normals>::value_type,
        Normals,
        HR
        > visitor(
            normals,
            hr
            );

    return apply_visitor(visitor, prim);
}

template <
    typename Normals,
    typename HR,
    typename ...Ts,
    typename NormalBinding
    >
VSNRAY_FUNC
inline auto get_shading_normal(
        Normals                     normals,
        HR const&                   hr,
        generic_primitive<Ts...>    prim,
        NormalBinding               /* */
        )
    -> typename std::iterator_traits<Normals>::value_type
{
    detail::get_shading_normal_from_generic_primitive_visitor<
        typename std::iterator_traits<Normals>::value_type,
        NormalBinding,
        Normals,
        HR
        > visitor(
            normals,
            hr
            );

    return apply_visitor(visitor, prim);
}

} // visionaray
