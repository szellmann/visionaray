// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Public interface
//

template <typename T>
VSNRAY_FUNC
inline spectrum<T> disney<T>::ambient() const
{
    return spectrum<T>(0.0);
}

template <typename T>
template <typename SR>
VSNRAY_FUNC
inline spectrum<typename SR::scalar_type> disney<T>::shade(SR const& sr) const
{
    using U = typename SR::scalar_type;

    auto wi = sr.light_dir;
    auto wo = sr.view_dir;
    auto n = sr.normal;
#if 1 // two-sided
    n = faceforward( n, sr.view_dir, sr.geometric_normal );
#endif
    auto ndotl = max( U(0.0), dot(n, wi) );

    spectrum<U> cd = from_rgb(sr.tex_color) * brdf_.f(n, wo, wi);

    return cd * constants::pi<U>() * from_rgb(sr.light_intensity) * ndotl;
}

template <typename T>
template <typename SR, typename U, typename Interaction, typename Generator>
VSNRAY_FUNC
inline spectrum<U> disney<T>::sample(
        SR const&       shade_rec,
        vector<3, U>&   refl_dir,
        U&              pdf,
        Interaction&    inter,
        Generator&      gen
        ) const
{
    auto n = shade_rec.normal;
#if 1 // two-sided
    n = faceforward( n, shade_rec.view_dir, shade_rec.geometric_normal );
#endif
    return from_rgb(shade_rec.tex_color) * brdf_.sample_f(n, shade_rec.view_dir, refl_dir, pdf, inter, gen);
}

template <typename T>
template <typename SR, typename Interaction> 
VSNRAY_FUNC
inline typename SR::scalar_type disney<T>::pdf(SR const& sr, Interaction const& inter) const
{
    VSNRAY_UNUSED(inter);

    auto n = sr.normal;
#if 1 // two-sided
    n = faceforward( n, sr.view_dir, sr.geometric_normal );
#endif
    return brdf_.pdf(n, sr.view_dir, sr.light_dir);
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& disney<T>::base_color()
{
    return brdf_.base_color;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> const& disney<T>::base_color() const
{
    return brdf_.base_color;
}

template <typename T>
VSNRAY_FUNC
inline T& disney<T>::sheen()
{
    return brdf_.sheen;
}

template <typename T>
VSNRAY_FUNC
inline T const& disney<T>::sheen() const
{
    return brdf_.sheen;
}

template <typename T>
VSNRAY_FUNC
inline T& disney<T>::sheen_tint()
{
    return brdf_.sheen_tint;
}

template <typename T>
VSNRAY_FUNC
inline T const& disney<T>::sheen_tint() const
{
    return brdf_.sheen_tint;
}

template <typename T>
VSNRAY_FUNC
inline T& disney<T>::roughness()
{
    return brdf_.roughness;
}

template <typename T>
VSNRAY_FUNC
inline T const& disney<T>::roughness() const
{
    return brdf_.roughness;
}

} // visionaray
