// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once 

#ifndef VSNRAY_PHASE_FUNCTION_H
#define VSNRAY_PHASE_FUNCTION_H 1

#include "math/constants.h"
#include "math/limits.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Henyey-Greenstein phase function
//

template <typename T>
class henyey_greenstein
{
public:

    using scalar_type = T;

public:

    scalar_type g;

    template <typename U>
    VSNRAY_FUNC
    U tr(vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        // TODO: calculate w/ high precision and add to math/constants.h
        static const U one_over_four_pi = U(1.0) / (U(4.0) * constants::pi<U>());

        U cost = dot(wo, wi);

        U denom = U(1.0) + U(g) * U(g) + U(2.0) * U(g) * cost;
        return one_over_four_pi * (U(1.0) - U(g) * U(g)) / (denom * sqrt(denom));
    }

    template <typename U, typename Generator>
    VSNRAY_FUNC
    U sample(vector<3, U> const& wo, vector<3, U>& wi, U& pdf, Generator& gen) const
    {
        auto g_not_zero = abs(U(g)) >= numeric_limits<U>::epsilon();

        U u1 = gen.next();
        U u2 = gen.next();

        U a = select(
            g_not_zero,
            (U(1.0) - U(g) * U(g)) / (U(1.0) - U(g) + U(2.0) * U(g) * u1),
            U(0.0)
            );

        U cost = select(
            g_not_zero,
            (U(1.0) + U(g) * U(g) - (a * a)) / (U(2.0) * U(g)),
            U(1.0) - U(2.0) * u1
            );

        U sint = sqrt(max(U(0.0), U(1.0) - cost * cost));
        U phi = constants::two_pi<U>() * u2;

        vector<3, U> u;
        vector<3, U> v;
        vector<3, U> w = wo;
        make_orthonormal_basis(u, v, w);

        wi = sint * cos(phi) * u + sint * sin(phi) * v + cost * -w;
        pdf = U(1.0);

        return tr(-w, wi);
    }
};

} // visionaray

#endif // VSNRAY_PHASE_FUNCTION
