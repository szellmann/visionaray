// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// spot_light members
//

template <typename T>
template <typename U>
VSNRAY_FUNC
inline vector<3, U> spot_light<T>::intensity(vector<3, U> const& pos) const
{
    using V = vector<3, U>;

    U att(1.0);

    V light_dir = V(position_) - pos;

#if 1 // use attenuation
    auto dist = length(light_dir);
    att = U(
        1.0 / (constant_attenuation_
             + linear_attenuation_    * dist
             + quadratic_attenuation_ * dist * dist)
        );
#endif

    auto spot = dot(V(spot_direction_), normalize(-light_dir));

    spot = select(
            spot > U(spot_cos_cutoff_),
            pow(spot, U(spot_exponent_)),
            U(0.0)
            );

    return vector<3, U>(cl_ * kl_) * att * spot;
}

template <typename T>
template <typename U, typename Generator>
VSNRAY_FUNC
inline vector<3, U> spot_light<T>::sample(U& pdf, Generator& gen) const
{
    VSNRAY_UNUSED(gen);

    pdf = U(1.0);
    return vector<3, U>(position());
}

template <typename T>
template <typename U, size_t N, typename Generator>
VSNRAY_FUNC
inline void spot_light<T>::sample(
        array<U, N>& pdfs,
        array<vector<3, U>, N>& result,
        Generator& gen
        ) const
{
    for (size_t i = 0; i < N; ++i)
    {
        result[i] = sample(pdfs[i], gen);
    }
}

template <typename T>
VSNRAY_FUNC
inline vector<3, T> spot_light<T>::position() const
{
    return position_;
}

template <typename T>
VSNRAY_FUNC
inline vector<3, T> spot_light<T>::spot_direction() const
{
    return spot_direction_;
}

template <typename T>
VSNRAY_FUNC
inline T spot_light<T>::spot_cutoff() const
{
    return spot_cutoff_;
}

template <typename T>
VSNRAY_FUNC
inline T spot_light<T>::spot_exponent() const
{
    return spot_exponent_;
}

template <typename T>
VSNRAY_FUNC
inline T spot_light<T>::constant_attenuation() const
{
    return constant_attenuation_;
}

template <typename T>
VSNRAY_FUNC
inline T spot_light<T>::linear_attenuation() const
{
    return linear_attenuation_;
}

template <typename T>
VSNRAY_FUNC
inline T spot_light<T>::quadratic_attenuation() const
{
    return quadratic_attenuation_;
}

template <typename T>
VSNRAY_FUNC
inline void spot_light<T>::set_cl(vector<3, T> const& cl)
{
    cl_ = cl;
}

template <typename T>
VSNRAY_FUNC
inline void spot_light<T>::set_kl(T kl)
{
    kl_ = kl;
}

template <typename T>
VSNRAY_FUNC
inline void spot_light<T>::set_position(vector<3, T> const& pos)
{
    position_ = pos;
}

template <typename T>
VSNRAY_FUNC
inline void spot_light<T>::set_spot_direction(vector<3, T> const& dir)
{
    spot_direction_ = dir;
}

template <typename T>
VSNRAY_FUNC
inline void spot_light<T>::set_spot_cutoff(T cutoff)
{
    spot_cutoff_     = cutoff;
    spot_cos_cutoff_ = cos(cutoff);
}

template <typename T>
VSNRAY_FUNC
inline void spot_light<T>::set_spot_exponent(T exponent)
{
    spot_exponent_ = exponent;
}

template <typename T>
VSNRAY_FUNC
inline void spot_light<T>::set_constant_attenuation(T att)
{
    constant_attenuation_ = att;
}

template <typename T>
VSNRAY_FUNC
inline void spot_light<T>::set_linear_attenuation(T att)
{
    linear_attenuation_ = att;
}

template <typename T>
VSNRAY_FUNC
inline void spot_light<T>::set_quadratic_attenuation(T att)
{
    quadratic_attenuation_ = att;
}

} // visionaray
