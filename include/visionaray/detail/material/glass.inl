// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Public interface
//

template <typename T>
VSNRAY_FUNC
inline spectrum<T> glass<T>::ambient() const
{
    return spectrum<T>(0.0); // TODO: no support for  ambient
}

template <typename T>
template <typename SR>
VSNRAY_FUNC
inline spectrum<typename SR::scalar_type> glass<T>::shade(SR const& sr) const
{
    return specular_bsdf_.f(sr.normal, sr.view_dir, sr.light_dir);
}

template <typename T>
template <typename SR, typename U, typename Interaction, typename Generator>
VSNRAY_FUNC
inline spectrum<U> glass<T>::sample(
        SR const&       sr,
        vector<3, U>&   refl_dir,
        U&              pdf,
        Interaction&    inter,
        Generator&      gen
        ) const
{
    auto f = specular_bsdf_.sample_f(sr.normal, sr.view_dir, refl_dir, pdf, inter, gen);
    // If ray originates in 2nd medium, flip normal
    auto n = select(
        U(dot(sr.normal, sr.view_dir)) > U(0.0),
        vector<3, U>(sr.normal),
        vector<3, U>(-sr.normal)
        );
    return f * (dot(n, refl_dir) / pdf);
}

template <typename T>
template <typename SR, typename Interaction>
VSNRAY_FUNC
inline typename SR::scalar_type glass<T>::pdf(SR const& sr, Interaction const& inter) const
{
    VSNRAY_UNUSED(inter);
    return specular_bsdf_.pdf(sr.normal, sr.view_dir, sr.light_dir);
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& glass<T>::ct()
{
    return specular_bsdf_.ct;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> const& glass<T>::ct() const
{
    return specular_bsdf_.ct;
}

template <typename T>
VSNRAY_FUNC
inline T& glass<T>::kt()
{
    return specular_bsdf_.kt;
}

template <typename T>
VSNRAY_FUNC
inline T const& glass<T>::kt() const
{
    return specular_bsdf_.kt;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& glass<T>::cr()
{
    return specular_bsdf_.cr;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> const& glass<T>::cr() const
{
    return specular_bsdf_.cr;
}

template <typename T>
VSNRAY_FUNC
inline T& glass<T>::kr()
{
    return specular_bsdf_.kr;
}

template <typename T>
VSNRAY_FUNC
inline T const& glass<T>::kr() const
{
    return specular_bsdf_.kr;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& glass<T>::ior()
{
    return specular_bsdf_.ior;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> const& glass<T>::ior() const
{
    return specular_bsdf_.ior;
}

} // visionaray
