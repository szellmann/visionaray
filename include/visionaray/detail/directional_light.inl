// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../math/constants.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// directional_light members
//

template <typename T>
template <typename U>
VSNRAY_FUNC
inline vector<3, U> directional_light<T>::intensity(vector<3, U> const& /*pos*/) const
{
    return vector<3, U>(cl_ * kl_);
}

template <typename T>
template <typename Generator, typename U>
VSNRAY_FUNC
inline light_sample<U> directional_light<T>::sample(vector<3, U> const& reference_point, Generator& gen) const
{
    light_sample<U> result;

    result.dist = numeric_limits<U>::max();
    vector<3, U> u, v;
    vector<3, U> directionU(direction_);
    make_orthonormal_basis(u, v, directionU);

    result.area = U(1.0);
    result.normal = normalize( vector<3, U>(
            gen.next() * U(2.0) - U(1.0),
            gen.next() * U(2.0) - U(1.0),
            gen.next() * U(2.0) - U(1.0)
            ) );

    const U cos_theta_max = cos(angular_diameter_ * constants::degrees_to_radians<T>());

    auto cone_sample = uniform_sample_cone(gen.next(), gen.next(), cos_theta_max);
    result.dir = select(
        cos_theta_max == U(0.0),
        directionU,
        u * cone_sample.x + v * cone_sample.y + directionU * cone_sample.z
        );

    result.pdf = select(
        cos_theta_max ==  U(0.0),
        U(1.0),
        cos_theta_max
        );

    return result;
}

template <typename T>
VSNRAY_FUNC
inline vector<3, T> directional_light<T>::direction() const
{
    return direction_;
}

template <typename T>
VSNRAY_FUNC
inline T directional_light<T>::angular_diameter() const
{
    return angular_diameter_;
}

template <typename T>
VSNRAY_FUNC
inline void directional_light<T>::set_cl(vector<3, T> const& cl)
{
    cl_ = cl;
}

template <typename T>
VSNRAY_FUNC
inline void directional_light<T>::set_kl(T kl)
{
    kl_ = kl;
}

template <typename T>
VSNRAY_FUNC
inline void directional_light<T>::set_direction(vector<3, T> const& dir)
{
    direction_ = dir;
}

template <typename T>
VSNRAY_FUNC
inline void directional_light<T>::set_angular_diameter(T const& ad)
{
    angular_diameter_ = ad;
}

} // visionaray
