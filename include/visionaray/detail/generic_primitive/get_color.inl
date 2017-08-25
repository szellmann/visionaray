// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/get_color.h>

namespace visionaray
{
namespace detail
{

template <typename ReturnType, typename Colors, typename ColorBinding, typename HR>
class get_color_from_generic_primitive_visitor
{
public:

    using return_type = ReturnType;

public:

    VSNRAY_FUNC
    get_color_from_generic_primitive_visitor(
            Colors      colors,
            HR const&   hr
            )
        : colors_(colors)
        , hr_(hr)
    {
    }

    template <typename Primitive>
    VSNRAY_FUNC
    return_type operator()(Primitive const& primitive) const
    {
        return get_color(colors_, hr_, primitive, ColorBinding{});
    }

private:

    Colors      colors_;
    HR const&   hr_;

};

} // detail


//-------------------------------------------------------------------------------------------------
// get_color() for generic primitives
//

template <
    typename Colors,
    typename HR,
    typename ...Ts,
    typename ColorBinding
    >
VSNRAY_FUNC
inline auto get_color(
        Colors                      colors,
        HR const&                   hr,
        generic_primitive<Ts...>    prim,
        ColorBinding                /* */
        )
    -> typename std::iterator_traits<Colors>::value_type
{
    detail::get_color_from_generic_primitive_visitor<
        typename std::iterator_traits<Colors>::value_type,
        Colors,
        HR,
        ColorBinding
        >visitor(
            colors,
            hr
            );

    return apply_visitor( visitor, prim );
}

} // visionaray
