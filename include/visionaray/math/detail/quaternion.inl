// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../matrix.h"
#include "math.h"

namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// quaternion members
//

template <typename T>
MATH_FUNC
inline quaternion<T>::quaternion(T const& w, T const& x, T const& y, T const& z)
    : w(w)
    , x(x)
    , y(y)
    , z(z)
{
}


template <typename T>
MATH_FUNC
inline quaternion<T>::quaternion(T const& w, vector<3, T> const& v)
    : w(w)
    , x(v.x)
    , y(v.y)
    , z(v.z)
{
}

template <typename T>
MATH_FUNC
inline quaternion<T> quaternion<T>::identity()
{
    return quaternion<T>(T(1.0), T(0.0), T(0.0), T(0.0));
}

template <typename T>
MATH_FUNC
inline quaternion<T> quaternion<T>::rotation(vector<3, T> const& from, vector<3, T> const& to)
{
    vector<3, T> nfrom = normalize(from);
    vector<3, T> nto   = normalize(to);

    return quaternion<T>(dot(nfrom, nto), cross(nfrom, nto));
}

template <typename T>
MATH_FUNC
inline quaternion<T> quaternion<T>::rotation(T const& yaw, T const& pitch, T const& roll)
{
    T cy = cos(yaw * T(0.5));
    T sy = sin(yaw * T(0.5));
    T cp = cos(pitch * T(0.5));
    T sp = sin(pitch * T(0.5));
    T cr = cos(roll * T(0.5));
    T sr = sin(roll * T(0.5));

    return quaternion<T>(
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr
        );
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template <typename T>
MATH_FUNC
inline quaternion<T> operator+(quaternion<T> const& p, quaternion<T> const& q)
{
    return quaternion<T>(p.w + q.w, p.x + q.x, p.y + q.y, p.z + q.z);
}

template <typename T>
MATH_FUNC
inline quaternion<T> operator*(quaternion<T> const& p, quaternion<T> const& q)
{
    return quaternion<T>(
        p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z,
        p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y,
        p.w * q.y - p.x * q.z + p.y * q.w + p.z * q.x,
        p.w * q.z + p.x * q.y - p.y * q.x + p.z * q.w
        );
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T>
MATH_FUNC
inline quaternion<T> operator*(quaternion<T> const& p, T const& s)
{
    return quaternion<T>(p.w * s, p.x * s, p.y * s, p.z * s);
}

template <typename T>
MATH_FUNC
inline quaternion<T> conjugate(quaternion<T> const& q)
{
    return quat(q.w, -q.x, -q.y, -q.z);
}

template <typename T>
MATH_FUNC
inline T dot(quaternion<T> const& p, quaternion<T> const& q)
{
    return p.w * q.w + p.x * q.x + p.y * q.y + p.z * q.z;
}

template <typename T>
MATH_FUNC
inline quaternion<T> inverse(quaternion<T> const& q)
{
    return conjugate(q) * (T(1.0) / dot(q, q));
}

template <typename T>
MATH_FUNC
inline T length(quaternion<T> const& q)
{
    return sqrt(dot(q, q));
}

template <typename T>
MATH_FUNC
inline quaternion<T> normalize(quaternion<T> const& q)
{
    return q * (T(1.0) / length(q));
}

template <typename T>
MATH_FUNC
inline quaternion<T> rotation(vector<3, T> const& axis, T const& angle)
{
    T s = sin(T(0.5) * angle) / length(axis);
    T c = cos(T(0.5) * angle);

    return quaternion<T>(c, s * axis[0], s * axis[1], s * axis[2]);
}

template <typename T>
MATH_FUNC
inline matrix<4, 4, T> rotation(quaternion<T> const& q)
{
    T xx = q.x * q.x;
    T xy = q.x * q.y;
    T xz = q.x * q.z;
    T xw = q.x * q.w;
    T yy = q.y * q.y;
    T yz = q.y * q.z;
    T yw = q.y * q.w;
    T zz = q.z * q.z;
    T zw = q.z * q.w;
    T ww = q.w * q.w;

    matrix<4, 4, T> result;

    result(0, 0) = T(2.0) * (ww + xx) - T(1.0);
    result(1, 0) = T(2.0) * (xy + zw);
    result(2, 0) = T(2.0) * (xz - yw);
    result(3, 0) = T(0.0);
    result(0, 1) = T(2.0) * (xy - zw);
    result(1, 1) = T(2.0) * (ww + yy) - T(1.0);
    result(2, 1) = T(2.0) * (yz + xw);
    result(3, 1) = T(0.0);
    result(0, 2) = T(2.0) * (xz + yw);
    result(1, 2) = T(2.0) * (yz - xw);
    result(2, 2) = T(2.0) * (ww + zz) - T(1.0);
    result(3, 2) = T(0.0);
    result(0, 3) = T(0.0);
    result(1, 3) = T(0.0);
    result(2, 3) = T(0.0);
    result(3, 3) = T(1.0);

    return result;
}

template <typename T>
MATH_FUNC
inline T rotation_angle(quaternion<T> const& q)
{
    return T(2.0) * acos(q.w);
}

template <typename T>
MATH_FUNC
inline vector<3, T> rotation_axis(quaternion<T> const& q)
{
    return normalize(vector<3, T>(q.x, q.y, q.z));
}

} // MATH_NAMESPACE
