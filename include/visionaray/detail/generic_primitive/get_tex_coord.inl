// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iterator>
#include <type_traits>

#include <visionaray/get_tex_coord.h>
#include <visionaray/prim_traits.h>

namespace visionaray
{
namespace detail
{

template <typename ReturnType, typename TexCoords, typename HR>
class get_tex_coord_from_generic_primitive_visitor
{
public:

    using return_type = ReturnType;

public:

    VSNRAY_FUNC
    get_tex_coord_from_generic_primitive_visitor(
            TexCoords   tex_coords,
            HR const&   hr
            )
        : tex_coords_(tex_coords)
        , hr_(hr)
    {
    }

    // primitive w/o precalculated texture coordinates
    template <typename Primitive>
    VSNRAY_FUNC
    return_type operator()(
            Primitive const& primitive,
            typename std::enable_if<num_tex_coords<Primitive>::value == 0>::type* = 0
            ) const
    {
        return get_tex_coord(hr_, primitive);
    }

    // primitive w/ precalculated texture coordinates
    template <typename Primitive>
    VSNRAY_FUNC
    return_type operator()(
            Primitive const& primitive,
            typename std::enable_if<num_tex_coords<Primitive>::value >= 1>::type* = 0
            ) const
    {
        return get_tex_coord(tex_coords_, hr_, primitive);
    }

private:

    TexCoords   tex_coords_;
    HR const&   hr_;

};

} // detail


//-------------------------------------------------------------------------------------------------
// get_tex_coord() for generic primitives
//

template <
    typename TexCoords,
    typename HR,
    typename ...Ts
    >
VSNRAY_FUNC
inline auto get_tex_coord(
        TexCoords                   tex_coords,
        HR const&                   hr,
        generic_primitive<Ts...>    prim
        )
    -> typename std::iterator_traits<TexCoords>::value_type
{
    detail::get_tex_coord_from_generic_primitive_visitor<
        typename std::iterator_traits<TexCoords>::value_type,
        TexCoords,
        HR
        >visitor(
            tex_coords,
            hr
            );

    return apply_visitor( visitor, prim );
}

} // visionaray
