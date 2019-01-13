// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <utility>

#include "../math/array.h"
#include "../generic_material.h"

namespace visionaray
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// SIMD conversions
//

template <typename N, typename C, typename M, size_t Size>
VSNRAY_FUNC
inline auto pack(array<surface<N, C, M>, Size> const& surfs)
    -> surface<
            decltype(pack(std::declval<array<N, Size>>())),
            decltype(pack(std::declval<array<C, Size>>())),
            decltype(pack(std::declval<array<M, Size>>()))
            >
{
    array<N, Size> geometric_normals;
    array<N, Size> shading_normals;
    array<C, Size> tex_colors;
    array<M, Size> materials;

    for (size_t i = 0; i < Size; ++i)
    {
        geometric_normals[i] = surfs[i].geometric_normal;
        shading_normals[i]   = surfs[i].shading_normal;
        tex_colors[i]        = surfs[i].tex_color;
        materials[i]         = surfs[i].material;
    }

    return {
        pack(geometric_normals),
        pack(shading_normals),
        pack(tex_colors),
        pack(materials)
        };
}

} // simd

} // visionaray
