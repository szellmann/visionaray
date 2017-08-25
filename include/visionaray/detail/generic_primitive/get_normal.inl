// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iterator>
#include <type_traits>

#include <visionaray/get_normal.h>
#include <visionaray/get_shading_normal.h>

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Call get_normal (or derivatives) for the underlying primitive
//

template <typename NormalFunc, typename ReturnType, typename NormalBinding, typename Normals, typename HR>
class get_normal_from_generic_primitive_visitor
{
public:

    using return_type = ReturnType;

public:

    VSNRAY_FUNC
    get_normal_from_generic_primitive_visitor(
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
            Primitive const& primitive,
            typename std::enable_if<num_normals<Primitive, NormalBinding>::value >= 2>::type* = 0
            ) const
    {
        NormalFunc t;
        return t(normals_, hr_, primitive, NormalBinding{});
    }

    // overload w/ precalculated normals - without the need for the primitive
    template <typename Primitive>
    VSNRAY_FUNC
    return_type operator()(
            Primitive const& primitive,
            typename std::enable_if<num_normals<Primitive, NormalBinding>::value == 1>::type* = 0
            ) const
    {
        VSNRAY_UNUSED(primitive);

        NormalFunc t;
        return t(normals_, hr_, Primitive{}, NormalBinding{});
    }

    // overload w/o normals
    template <typename Primitive>
    VSNRAY_FUNC
    return_type operator()(
            Primitive const& primitive,
            typename std::enable_if<num_normals<Primitive, NormalBinding>::value == 0>::type* = 0
            ) const
    {
        VSNRAY_UNUSED(primitive);

        NormalFunc t;
        return t(hr_, primitive);
    }

private:

    Normals     normals_;
    HR const&   hr_;

};

} // detail


//-------------------------------------------------------------------------------------------------
// get_normal overloads
//

template <
    typename Normals,
    typename HR,
    typename ...Ts,
    typename NormalBinding
    >
VSNRAY_FUNC
inline auto get_normal(
        Normals                     normals,
        HR const&                   hr,
        generic_primitive<Ts...>    prim,
        NormalBinding               /* */
        )
    -> typename std::iterator_traits<Normals>::value_type
{
    detail::get_normal_from_generic_primitive_visitor<
        detail::get_normal_t,
        typename std::iterator_traits<Normals>::value_type,
        NormalBinding,
        Normals,
        HR
        >visitor(
            normals,
            hr
            );

    return apply_visitor( visitor, prim );
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
    detail::get_normal_from_generic_primitive_visitor<
        detail::get_shading_normal_t,
        typename std::iterator_traits<Normals>::value_type,
        NormalBinding,
        Normals,
        HR
        >visitor(
            normals,
            hr
            );

    return apply_visitor( visitor, prim );
}

} // visionaray
