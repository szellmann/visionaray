// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{


//--------------------------------------------------------------------------------------------------
// plnae3 members
//

template <typename T>
inline basic_plane<3, T>::basic_plane()
{
}

template <typename T>
inline basic_plane<3, T>::basic_plane(vector<3, T> const& n, T o)
    : normal(n)
    , offset(o)
{
}

template <typename T>
inline basic_plane<3, T>::basic_plane(vector<3, T> const& n, vector<3, T> const& p)
    : normal(n)
    , offset(dot(n, p))
{
}

} // MATH_NAMESPACE


