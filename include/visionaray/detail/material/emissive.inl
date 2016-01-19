// This file is distributed under the MIT license.
// See the LICENSE file for details.

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
    using U = typename SR::scalar_type;
    return select(
            dot(sr.normal, sr.view_dir) >= U(0.0),
            ce(sr),
            spectrum<U>(0.0)
            );
}

template <typename T>
template <typename SR, typename U, typename Sampler>
VSNRAY_FUNC
inline spectrum<U> emissive<T>::sample(
        SR const&       shade_rec,
        vector<3, U>&   refl_dir,
        U&              pdf,
        Sampler&        sampler
        ) const
{
    VSNRAY_UNUSED(refl_dir); // TODO?
    VSNRAY_UNUSED(sampler);
    pdf = U(1.0);
    return shade(shade_rec);
}

template <typename T>
VSNRAY_FUNC
inline void emissive<T>::set_ce(spectrum<T> const& ce)
{
    ce_ = ce;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> emissive<T>::get_ce() const
{
    return ce_;
}

template <typename T>
VSNRAY_FUNC
inline void emissive<T>::set_ls(T ls)
{
    ls_ = ls;
}

template <typename T>
VSNRAY_FUNC
inline T emissive<T>::get_ls() const
{
    return ls_;
}


//-------------------------------------------------------------------------------------------------
// Private functions
//

template <typename T>
template <typename SR>
VSNRAY_FUNC
inline spectrum<T> emissive<T>::ce(SR const& sr) const
{
    VSNRAY_UNUSED(sr);
    return ce_ * ls_;
}

template <typename T>
template <typename L, typename C, typename S>
VSNRAY_FUNC
inline spectrum<T> emissive<T>::ce(shade_record<L, C, S> const& sr) const
{
    return spectrum<T>(from_rgb(sr.tex_color)) * ce_ * ls_;
}

} // visionaray
