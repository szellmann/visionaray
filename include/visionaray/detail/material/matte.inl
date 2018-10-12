// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/constants.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Public interface
//

template <typename T>
VSNRAY_FUNC
inline spectrum<T> matte<T>::ambient() const
{
    return ca_ * ka_;
}

template <typename T>
template <typename SR>
VSNRAY_FUNC
inline spectrum<typename SR::scalar_type> matte<T>::shade(SR const& sr) const
{
    using U = typename SR::scalar_type;

    auto wi = sr.light_dir;
    auto wo = sr.view_dir;
    auto n = sr.normal;
#if 1 // two-sided
    n = faceforward( n, sr.view_dir, sr.geometric_normal );
#endif
    auto ndotl = max( U(0.0), dot(n, wi) );

    spectrum<U> cd = from_rgb(sr.tex_color) * diffuse_brdf_.f(n, wo, wi);

    return cd * constants::pi<U>() * from_rgb(sr.light_intensity) * ndotl;
}

template <typename T>
template <typename SR, typename U, typename Interaction, typename Generator>
VSNRAY_FUNC
inline spectrum<U> matte<T>::sample(
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
    return from_rgb(shade_rec.tex_color) * diffuse_brdf_.sample_f(n, shade_rec.view_dir, refl_dir, pdf, inter, gen);
}

template <typename T>
template <typename SR, typename Interaction> 
VSNRAY_FUNC
inline typename SR::scalar_type matte<T>::pdf(SR const& sr, Interaction const& inter) const
{
    VSNRAY_UNUSED(inter);

    auto n = sr.normal;
#if 1 // two-sided
    n = faceforward( n, sr.view_dir, sr.geometric_normal );
#endif
    return diffuse_brdf_.pdf(n, sr.view_dir, sr.light_dir);
}

// --- deprecated begin -----------------------------------

template <typename T>
VSNRAY_FUNC
inline void matte<T>::set_ca(spectrum<T> const& ca)
{
    ca_ = ca;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> matte<T>::get_ca() const
{
    return ca_;
}

template <typename T>
VSNRAY_FUNC
inline void matte<T>::set_ka(T ka)
{
    ka_ = ka;
}

template <typename T>
VSNRAY_FUNC
inline T matte<T>::get_ka() const
{
    return ka_;
}

template <typename T>
VSNRAY_FUNC void matte<T>::set_cd(spectrum<T> const& cd)
{
    diffuse_brdf_.cd = cd;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> matte<T>::get_cd() const
{
    return diffuse_brdf_.cd;
}

template <typename T>
VSNRAY_FUNC
inline void matte<T>::set_kd(T kd)
{
    diffuse_brdf_.kd = kd;
}

template <typename T>
VSNRAY_FUNC
inline T matte<T>::get_kd() const
{
    return diffuse_brdf_.kd;
}

// --- deprecated end -------------------------------------

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& matte<T>::ca()
{
    return ca_;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> const& matte<T>::ca() const
{
    return ca_;
}

template <typename T>
VSNRAY_FUNC
inline T& matte<T>::ka()
{
    return ka_;
}

template <typename T>
VSNRAY_FUNC
inline T const& matte<T>::ka() const
{
    return ka_;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& matte<T>::cd()
{
    return diffuse_brdf_.cd;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> const& matte<T>::cd() const
{
    return diffuse_brdf_.cd;
}

template <typename T>
VSNRAY_FUNC
inline T& matte<T>::kd()
{
    return diffuse_brdf_.kd;
}

template <typename T>
VSNRAY_FUNC
inline T const& matte<T>::kd() const
{
    return diffuse_brdf_.kd;
}

} // visionaray
