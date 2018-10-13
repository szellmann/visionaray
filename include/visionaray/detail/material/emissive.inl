// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/surface_interaction.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Public interface
//

template <typename T>
VSNRAY_FUNC
inline spectrum<T> emissive<T>::ambient() const
{
    return spectrum<T>();
}

template <typename T>
template <typename SR>
VSNRAY_FUNC
inline spectrum<typename SR::scalar_type> emissive<T>::shade(SR const& sr) const
{
    return from_rgb(sr.tex_color) * ce_ * ls_;
}

template <typename T>
template <typename SR, typename U, typename Interaction, typename Generator>
VSNRAY_FUNC
inline spectrum<U> emissive<T>::sample(
        SR const&       shade_rec,
        vector<3, U>&   refl_dir,
        U&              pdf,
        Interaction&    inter,
        Generator&      gen
        ) const
{
    VSNRAY_UNUSED(refl_dir); // TODO?
    VSNRAY_UNUSED(gen);
    pdf = U(1.0);
    inter = Interaction(surface_interaction::Emission);
    return shade(shade_rec);
}

template <typename T>
template <typename SR, typename Interaction>
VSNRAY_FUNC
inline typename SR::scalar_type emissive<T>::pdf(SR const& sr, Interaction const& inter) const
{
    VSNRAY_UNUSED(sr, inter);
    return typename SR::scalar_type(1.0);
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& emissive<T>::ce()
{
    return ce_;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> const& emissive<T>::ce() const
{
    return ce_;
}

template <typename T>
VSNRAY_FUNC
inline T& emissive<T>::ls()
{
    return ls_;
}

template <typename T>
VSNRAY_FUNC
inline T const& emissive<T>::ls() const
{
    return ls_;
}

} // visionaray
