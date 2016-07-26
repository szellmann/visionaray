// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// point_light members
//

template <typename T>
template <typename U>
VSNRAY_FUNC
inline vector<3, U> point_light<T>::intensity(vector<3, U> const& pos) const
{
    U att(1.0);

#if 1 // use attenuation
    auto dist = length(vector<3, U>(position_) - pos);
    att = U(
        1.0 / (constant_attenuation_
             + linear_attenuation_    * dist
             + quadratic_attenuation_ * dist * dist)
        );
#endif

    return vector<3, U>(cl_ * kl_) * att;
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
VSNRAY_FUNC
inline void point_light<T>::set_cl(vector<3, T> const& cl)
{
    cl_ = cl;
}

template <typename T>
VSNRAY_FUNC
inline void point_light<T>::set_kl(T kl)
{
    kl_ = kl;
}

template <typename T>
VSNRAY_FUNC
inline void point_light<T>::set_position(vector<3, T> const& pos)
{
    position_ = pos;
}

template <typename t>
VSNRAY_FUNC
inline void point_light<t>::set_constant_attenuation(t att)
{
    constant_attenuation_ = att;
}

template <typename t>
VSNRAY_FUNC
inline void point_light<t>::set_linear_attenuation(t att)
{
    linear_attenuation_ = att;
}

template <typename t>
VSNRAY_FUNC
inline void point_light<t>::set_quadratic_attenuation(t att)
{
    quadratic_attenuation_ = att;
}

} // visionaray
