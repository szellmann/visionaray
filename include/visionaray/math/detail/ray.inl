// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{

template <typename T>
MATH_FUNC
inline basic_ray<T>::basic_ray(vector<3, T> const& o, vector<3, T> const& d)
    : ori(o)
    , dir(d)
{
}

} // MATH_NAMESPACE
