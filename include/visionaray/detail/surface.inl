// This file is distributed under the MIT license.
// See the LICENSE file for details.

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

} // visionaray
