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
VSNRAY_FUNC void ambient_light<T>::set_cl(vector<3, T> const& cl)
{
    cl_ = cl;
}

template <typename T>
VSNRAY_FUNC void ambient_light<T>::set_kl(T const& kl)
{
    kl_ = kl;
}

} // visionaray
