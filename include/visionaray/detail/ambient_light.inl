// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

template <typename T>
template <typename U>
VSNRAY_FUNC vector<3, U> ambient_light<T>::intensity(vector<3, U> const& /*dir*/) const
{
    return vector<3, U>(cl_) * U(kl_);
}

template <typename T>
template <typename U>
VSNRAY_FUNC vector<3, U> ambient_light<T>::background_intensity(vector<3, U> const& /*dir*/) const
{
    return vector<3, U>(background_cl_) * U(background_kl_);
}

template <typename T>
VSNRAY_FUNC void ambient_light<T>::set_cl(vector<3, T> const& cl)
{
    cl_ = cl;
}

template <typename T>
VSNRAY_FUNC void ambient_light<T>::set_kl(T const& kl)
{
    kl_ = kl;
}

template <typename T>
VSNRAY_FUNC void ambient_light<T>::set_background_cl(vector<3, T> const& cl)
{
    background_cl_ = cl;
}

template <typename T>
VSNRAY_FUNC void ambient_light<T>::set_background_kl(T const& kl)
{
    background_kl_ = kl;
}

} // visionaray
