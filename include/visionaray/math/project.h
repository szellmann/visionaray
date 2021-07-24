// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_PROJECT_H
#define VSNRAY_MATH_PROJECT_H 1

#include "config.h"
#include "matrix.h"
#include "rectangle.h"
#include "vector.h"

namespace MATH_NAMESPACE
{

template <typename T>
MATH_FUNC
inline void project(
        vector<3, T>&           win,
        vector<3, T> const&     obj,
        matrix<4, 4, T> const&  modelview,
        matrix<4, 4, T> const&  projection,
        recti const&            viewport
        )
{
    auto tmp = projection * modelview * vector<4, T>(obj, T(1.0));

    auto v = tmp.xyz() / tmp.w;

    win[0] = viewport[0] + viewport[2] * (v[0] + 1) / T(2.0);
    win[1] = viewport[1] + viewport[3] * (v[1] + 1) / T(2.0);
    win[2] = (v[2] + 1) / T(2.0);
}

template <typename T>
MATH_FUNC
inline void unproject(
        vector<3, T>&           obj,
        vector<3, T> const&     win,
        matrix<4, 4, T> const&  modelview,
        matrix<4, 4, T> const&  projection,
        recti const&            viewport
        )
{
    vector<4, T> u(
            T(2.0 * (win[0] - viewport[0]) / viewport[2] - 1.0),
            T(2.0 * (win[1] - viewport[1]) / viewport[3] - 1.0),
            T(2.0 * win[2] - 1.0),
            T(1.0)
            );

    auto invpm = inverse( projection * modelview );

    auto v = invpm * u;

    obj = v.xyz() / v.w;
}

} // MATH_NAMESPACE

#endif // VSNRAY_MATH_PROJECT_H
