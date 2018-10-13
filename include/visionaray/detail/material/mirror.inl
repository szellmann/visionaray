// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Public interface
//

template <typename T>
VSNRAY_FUNC
inline spectrum<T> mirror<T>::ambient() const
{
    return spectrum<T>(0.0); // TODO: no support for  ambient
}

template <typename T>
template <typename SR>
VSNRAY_FUNC
inline spectrum<typename SR::scalar_type> mirror<T>::shade(SR const& sr) const
{
    auto n = sr.normal;
#if 1 // two-sided
    n = faceforward( n, sr.view_dir, sr.geometric_normal );
#endif
    return specular_brdf_.f(n, sr.view_dir, sr.light_dir);
}

template <typename T>
template <typename SR, typename U, typename Interaction, typename Generator>
VSNRAY_FUNC
inline spectrum<U> mirror<T>::sample(
        SR const&       sr,
        vector<3, U>&   refl_dir,
        U&              pdf,
        Interaction&    inter,
        Generator&      gen
        ) const
{
    return specular_brdf_.sample_f(sr.normal, sr.view_dir, refl_dir, pdf, inter, gen);
}

template <typename T>
template <typename SR, typename Interaction> 
VSNRAY_FUNC
inline typename SR::scalar_type mirror<T>::pdf(SR const& sr, Interaction const& inter) const
{
    VSNRAY_UNUSED(inter);

    auto n = sr.normal;
#if 1 // two-sided
    n = faceforward( n, sr.view_dir, sr.geometric_normal );
#endif
    return specular_brdf_.pdf(n, sr.view_dir, sr.light_dir);
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& mirror<T>::cr()
{
    return specular_brdf_.cr;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> const& mirror<T>::cr() const
{
    return specular_brdf_.cr;
}

template <typename T>
VSNRAY_FUNC
inline T& mirror<T>::kr()
{
    return specular_brdf_.kr;
}

template <typename T>
VSNRAY_FUNC
inline T const& mirror<T>::kr() const
{
    return specular_brdf_.kr;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& mirror<T>::ior()
{
    return specular_brdf_.ior;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> const& mirror<T>::ior() const
{
    return specular_brdf_.ior;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& mirror<T>::absorption()
{
    return specular_brdf_.absorption;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> const& mirror<T>::absorption() const
{
    return specular_brdf_.absorption;
}

} // visionaray
