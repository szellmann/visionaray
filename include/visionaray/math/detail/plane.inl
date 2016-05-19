// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// plane members
//

template <size_t Dim, typename T, typename P>
MATH_FUNC
inline basic_plane<Dim, T, P>::basic_plane(vector<Dim, T> const& n, T o)
    : normal(n)
    , offset(o)
{
}

template <size_t Dim, typename T, typename P>
MATH_FUNC
inline basic_plane<Dim, T, P>::basic_plane(vector<Dim, T> const& n, vector<Dim, T> const& p)
    : normal(n)
    , offset(dot(n, p))
{
}


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template <size_t Dim, typename T, typename P>
MATH_FUNC
inline bool operator==(basic_plane<Dim, T, P> const& a, basic_plane<Dim, T, P> const& b)
{
    return a.normal == b.normal && a.offset == b.offset;
}

template <size_t Dim, typename T, typename P>
MATH_FUNC
inline bool operator!=(basic_plane<Dim, T, P> const& a, basic_plane<Dim, T, P> const& b)
{
    return a.normal != b.normal || a.offset != b.offset;
}

} // MATH_NAMESPACE
