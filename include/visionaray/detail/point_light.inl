// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// point_light members
//

template <typename T>
VSNRAY_FUNC
inline vector<3, T> point_light<T>::color() const
{
    return cl_ * kl_;
}

template <typename T>
VSNRAY_FUNC
inline vector<3, T> point_light<T>::position() const
{
    return position_;
}

template <typename T>
VSNRAY_FUNC
inline T point_light<T>::constant_attenuation() const
{
    return constant_attenuation_;
}

template <typename T>
VSNRAY_FUNC
inline T point_light<T>::linear_attenuation() const
{
    return linear_attenuation_;
}

template <typename T>
VSNRAY_FUNC
inline T point_light<T>::quadratic_attenuation() const
{
    return quadratic_attenuation_;
}

template <typename T>
inline void point_light<T>::set_cl(vector<3, T> const& cl)
{
    cl_ = cl;
}

template <typename T>
inline void point_light<T>::set_kl(T kl)
{
    kl_ = kl;
}

template <typename T>
inline void point_light<T>::set_position(vector<3, T> const& pos)
{
    position_ = pos;
}

template <typename t>
inline void point_light<t>::set_constant_attenuation(t att)
{
    constant_attenuation_ = att;
}

template <typename t>
inline void point_light<t>::set_linear_attenuation(t att)
{
    linear_attenuation_ = att;
}

template <typename t>
inline void point_light<t>::set_quadratic_attenuation(t att)
{
    quadratic_attenuation_ = att;
}

} // visionaray
