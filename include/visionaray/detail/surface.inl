// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>
#include <cstddef>
#include <utility>

#include "../generic_material.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// General surface functions
//

template <typename ...Args>
VSNRAY_FUNC
inline bool has_emissive_material(surface<Args...> const& surf)
{
    VSNRAY_UNUSED(surf);
    return false;
}

template <typename N, typename T>
VSNRAY_FUNC
inline bool has_emissive_material(surface<N, emissive<T>> const& surf)
{
    VSNRAY_UNUSED(surf);
    return true;
}

template <typename N, typename ...Ms, typename ...Ts>
VSNRAY_FUNC
inline auto has_emissive_material(surface<N, generic_material<Ms...>, Ts...> const& surf)
    -> decltype( surf.material.is_emissive() )
{
    return surf.material.is_emissive();
}

template <typename N, typename ...Ms, typename ...Ts>
VSNRAY_FUNC
inline auto has_emissive_material(surface<N, simd::generic_material4<Ms...>, Ts...> const& surf)
    -> decltype( surf.material.is_emissive() )
{
    return surf.material.is_emissive();
}


//-------------------------------------------------------------------------------------------------
// Factory function make_surface()
//

template <typename N, typename M>
VSNRAY_FUNC
inline surface<N, M> make_surface(N const& gn, N const& sn, M const& m)
{
    return surface<N, M>(gn, sn, m);
}

template <typename N, typename M, typename C>
VSNRAY_FUNC
inline surface<N, M, C> make_surface(N const& gn, N const& sn, M const m, C const& tex_color)
{
    return surface<N, M, C>(gn, sn, m, tex_color);
}


namespace simd
{

//-------------------------------------------------------------------------------------------------
// Functions to pack and unpack SIMD surfaces
//

template <typename N, typename M, typename ...Args, size_t Size>
inline auto pack(std::array<surface<N, M, Args...>, Size> const& surfs)
    -> decltype( make_surface(
            pack(std::declval<std::array<N, Size>>()),
            pack(std::declval<std::array<N, Size>>()),
            pack(std::declval<std::array<M, Size>>())
            ) )
{
    std::array<N, Size> geometric_normals;
    std::array<N, Size> shading_normals;
    std::array<M, Size> materials;

    for (size_t i = 0; i < Size; ++i)
    {
        geometric_normals[i] = surfs[i].geometric_normal;
        shading_normals[i]   = surfs[i].shading_normal;
        materials[i]         = surfs[i].material;
    }

    return make_surface(
            pack(geometric_normals),
            pack(shading_normals),
            pack(materials)
            );
}

template <typename N, typename M, typename C, typename ...Args, size_t Size>
inline auto pack(std::array<surface<N, M, C, Args...>, Size> const& surfs)
    -> decltype( make_surface(
            pack(std::declval<std::array<N, Size>>()),
            pack(std::declval<std::array<N, Size>>()),
            pack(std::declval<std::array<M, Size>>()),
            pack(std::declval<std::array<C, Size>>())
            ) )
{
    std::array<N, Size> geometric_normals;
    std::array<N, Size> shading_normals;
    std::array<M, Size> materials;
    std::array<C, Size> tex_colors;

    for (size_t i = 0; i < Size; ++i)
    {
        geometric_normals[i] = surfs[i].geometric_normal;
        shading_normals[i]   = surfs[i].shading_normal;
        materials[i]         = surfs[i].material;
        tex_colors[i]        = surfs[i].tex_color_;
    }

    return make_surface(
            pack(geometric_normals),
            pack(shading_normals),
            pack(materials),
            pack(tex_colors)
            );
}

} // simd

} // visionaray
