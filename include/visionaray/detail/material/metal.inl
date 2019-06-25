// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Public interface
//

template <typename T>
VSNRAY_FUNC
inline spectrum<T> metal<T>::ambient() const
{
    return spectrum<T>(0.0); // TODO: no support for  ambient
}

template <typename T>
template <typename SR>
VSNRAY_FUNC
inline spectrum<typename SR::scalar_type> metal<T>::shade(SR const& sr) const
{
    auto n = sr.normal;
#if 1 // two-sided
    n = faceforward( n, sr.view_dir, sr.geometric_normal );
#endif
    return brdf_.f(n, sr.view_dir, sr.light_dir);
}

template <typename T>
template <typename SR, typename U, typename Interaction, typename Generator>
VSNRAY_FUNC
inline spectrum<U> metal<T>::sample(
        SR const&       sr,
        vector<3, U>&   refl_dir,
        U&              pdf,
        Interaction&    inter,
        Generator&      gen
        ) const
{
    return brdf_.sample_f(sr.normal, sr.view_dir, refl_dir, pdf, inter, gen);
}

template <typename T>
template <typename SR, typename Interaction> 
VSNRAY_FUNC
inline typename SR::scalar_type metal<T>::pdf(SR const& sr, Interaction const& inter) const
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
inline T& metal<T>::roughness()
{
    return brdf_.mdf.alpha;
}

template <typename T>
VSNRAY_FUNC
inline T const& metal<T>::roughness() const
{
    return brdf_.mdf.alpha;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& metal<T>::ior()
{
    return brdf_.ior;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> const& metal<T>::ior() const
{
    return brdf_.ior;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T>& metal<T>::absorption()
{
    return brdf_.absorption;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> const& metal<T>::absorption() const
{
    return brdf_.absorption;
}

} // visionaray
