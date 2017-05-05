// This file is distributed under the MIT license.
// See the LICENSE file for details.

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

    spectrum<U> result(0.0);

    auto l = sr.light;
    auto wi = sr.light_dir;
    auto wo = sr.view_dir;
    auto n = sr.normal;
    auto ndotl = max( U(0.0), dot(n, wi) );

    return spectrum<U>(
            constants::pi<U>()
          * cd_impl(sr, n, wo, wi)
          * spectrum<U>(from_rgb(l.intensity(sr.isect_pos)))
          * ndotl
            );
}

template <typename T>
template <typename SR, typename U, typename Sampler>
VSNRAY_FUNC
inline spectrum<U> matte<T>::sample(
        SR const&       shade_rec,
        vector<3, U>&   refl_dir,
        U&              pdf,
        Sampler&        sampler
        ) const
{
    return sample_impl(shade_rec, refl_dir, pdf, sampler);
}

template <typename T>
template <typename L, typename C, typename U, typename Sampler>
VSNRAY_FUNC
inline spectrum<U> matte<T>::sample(
        shade_record<L, C, U> const&    sr,
        vector<3, U>&                   refl_dir,
        U&                              pdf,
        Sampler&                        sampler
        ) const
{
    return spectrum<U>(from_rgb(sr.tex_color)) * sample_impl(sr, refl_dir, pdf, sampler);
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


//-------------------------------------------------------------------------------------------------
// Private functions
//

template <typename T>
template <typename SR, typename V>
VSNRAY_FUNC
inline spectrum<T> matte<T>::cd_impl(SR const& sr, V const& n, V const& wo, V const& wi) const
{
    VSNRAY_UNUSED(sr);
    return diffuse_brdf_.f(n, wo, wi);
}

template <typename T>
template <typename L, typename C, typename S, typename V>
VSNRAY_FUNC
inline spectrum<T> matte<T>::cd_impl(shade_record<L, C, S> const& sr, V const& n, V const& wo, V const& wi) const
{
    return spectrum<T>(from_rgb(sr.tex_color)) * diffuse_brdf_.f(n, wo, wi);
}

template <typename T>
template <typename SR, typename U, typename Sampler>
VSNRAY_FUNC
inline spectrum<U> matte<T>::sample_impl(
        SR const&       shade_rec,
        vector<3, U>&   refl_dir,
        U&              pdf,
        Sampler&        sampler
        ) const
{
    return diffuse_brdf_.sample_f(shade_rec.normal, shade_rec.view_dir, refl_dir, pdf, sampler);
}

} // visionaray
