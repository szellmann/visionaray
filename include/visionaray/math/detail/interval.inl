// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{

template <typename T>
MATH_FUNC
inline interval<T>::interval(T const& t)
    : min(t)
    , max(t)
{
}

template <typename T>
MATH_FUNC
inline interval<T>::interval(T const& lo, T const& up)
    : min(lo)
    , max(up)
{
}

template <typename T>
MATH_FUNC
inline interval<T>& interval<T>::extend(T const& t)
{
    min = visionaray::min(min, t);
    max = visionaray::max(max, t);
    return *this;
}

template <typename T>
MATH_FUNC
inline interval<T>& interval<T>::extend(interval<T> const& t)
{
    min = visionaray::min(min, t.min);
    max = visionaray::max(max, t.max);
    return *this;
}

template <typename T>
MATH_FUNC
inline bool interval<T>::contains(T const& t) const
{
    return t >= min && t <= max;
}

template <typename T>
MATH_FUNC
inline bool operator==(interval<T> const& a, interval<T> const& b)
{
    return a.min == b.min && a.max == b.max;
}

} // MATH_NAMESPACE
