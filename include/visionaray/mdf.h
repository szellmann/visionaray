// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MDF_H
#define VSNRAY_MDF_H 1

#include "math/constants.h"
#include "math/vector.h"

namespace visionaray
{

template <typename T>
class ggx
{
public:
    using scalar_type = T;

public:

    T alpha;

    // Return differential area of microfacet w.r.t. h
    template <typename U>
    U d(vector<3, U> const& n, vector<3, U> const& h) const
    {
        U sgn = select(dot(n, h) > U(0.0), U(1.0), U(0.0));

        U theta_h = dot(n, h);
        U cos4 = cos(theta_h) * cos(theta_h) * cos(theta_h) * cos(theta_h);
        U tan2 = tan(theta_h) * tan(theta_h);

        return                          (alpha * alpha * sgn)
            / (constants::pi<U>() * cos4 * (alpha * alpha + tan2) * (alpha * alpha + tan2));
    }

    // Return _monodirectional_ shadowing-masking term as in Smith approximation
    template <typename U>
    U g1(vector<3, U> const& n, vector<3, U> const& h, vector<3, U> const& w) const
    {
        U sgn = select(dot(w, h) / dot(w, n) > U(0.0), U(1.0), U(0.0));

        U theta_v = dot(n, w);
        U tan2 = tan(theta_v) * tan(theta_v);

        return                  sgn * U(2.0)
            / (U(1.0) + sqrt(U(1.0) + alpha * alpha * tan2));
    }
};

} // visionaray

#endif // VSNRAY_MDF_H
