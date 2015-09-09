// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../matrix.h"
#include "math.h"

namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// quat members
//

inline quat::quat()
{
}

inline quat::quat(float w, float x, float y, float z)
    : w(w)
    , x(x)
    , y(y)
    , z(z)
{
}


inline quat::quat(float w, vec3 const& v)
    : w(w)
    , x(v.x)
    , y(v.y)
    , z(v.z)
{
}

inline quat quat::identity()
{
    return quat(1.0f, 0.0f, 0.0f, 0.0f);
}

inline quat quat::rotation(vec3 const& from, vec3 const& to)
{
    vec3 nfrom = normalize(from);
    vec3 nto   = normalize(to);

    return quat(dot(nfrom, nto), cross(nfrom, nto));
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

inline quat operator+(quat const& p, quat const& q)
{
    return quat(p.w + q.w, p.x + q.x, p.y + q.y, p.z + q.z);
}

inline quat operator*(quat const& p, quat const& q)
{
    return quat(
        p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z,
        p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y,
        p.w * q.y - p.x * q.z + p.y * q.w + p.z * q.x,
        p.w * q.z + p.x * q.y - p.y * q.x + p.z * q.w
        );
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

inline quat operator*(quat const& p, float s)
{
    return quat(p.w * s, p.x * s, p.y * s, p.z * s);
}

inline quat conjugate(quat const& q)
{
    return quat(q.w, -q.x, -q.y, -q.z);
}

inline float dot(quat const& p, quat const& q)
{
    return p.w * q.w + p.x * q.x + p.y * q.y + p.z * q.z;
}

inline quat inverse(quat const& q)
{
    return conjugate(q) * (1.0f / dot(q, q));
}

inline float length(quat const& q)
{
    return sqrt(dot(q, q));
}

inline quat normalize(quat const& q)
{
    return q * (1.0f / length(q));
}

inline quat rotation(vec3 const& axis, float angle)
{
    float s = sin(0.5f * angle) / length(axis);
    float c = cos(0.5f * angle);

    return quat(c, s * axis[0], s * axis[1], s * axis[2]);
}

inline mat4 rotation(quat const& q)
{
    float xx = q.x * q.x;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float xw = q.x * q.w;
    float yy = q.y * q.y;
    float yz = q.y * q.z;
    float yw = q.y * q.w;
    float zz = q.z * q.z;
    float zw = q.z * q.w;
    float ww = q.w * q.w;

    mat4 result;

    result(0, 0) = 2.0f * (ww + xx) - 1.0f;
    result(1, 0) = 2.0f * (xy + zw);
    result(2, 0) = 2.0f * (xz - yw);
    result(3, 0) = 0;
    result(0, 1) = 2.0f * (xy - zw);
    result(1, 1) = 2.0f * (ww + yy) - 1.0f;
    result(2, 1) = 2.0f * (yz + xw);
    result(3, 1) = 0;
    result(0, 2) = 2.0f * (xz + yw);
    result(1, 2) = 2.0f * (yz - xw);
    result(2, 2) = 2.0f * (ww + zz) - 1.0f;
    result(3, 2) = 0;
    result(0, 3) = 0;
    result(1, 3) = 0;
    result(2, 3) = 0;
    result(3, 3) = 1;

    return result;
}

inline float rotation_angle(quat const& q)
{
    return 2.0f * acos(q.w);
}

inline vec3 rotation_axis(quat const& q)
{
    return normalize(vec3(q.x, q.y, q.z));
}

} // MATH_NAMESPACE
