// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SIMPLE_INL
#define VSNRAY_SIMPLE_INL

#include <visionaray/surface.h>

#include "traverse.h"


namespace visionaray
{
namespace simple
{

template <typename Params>
struct kernel
{

    Params params;

    template <typename R>
    VSNRAY_FUNC vector<4, typename R::scalar_type> operator()(R ray) const
    {
        using C = vector<4, typename R::scalar_type>;
        using S = typename R::scalar_type;

        auto hit_rec = closest_hit(ray, params.prims.begin, params.prims.end);

        if (any(hit_rec.hit))
        {
            auto surf = get_surface(hit_rec, params);
            auto ambient = C(surf.material.ambient(), S(1.0)) * C(params.ambient_color);
            C shaded_clr = select( hit_rec.hit, ambient, C(params.bg_color) );

            for (auto it = params.lights.begin; it != params.lights.end; ++it)
            {
                auto sr     = make_shade_record<Params, S>();
                sr.active   = hit_rec.hit;
                sr.view_dir = -ray.dir;
                sr.light    = it;
                auto clr    = surf.shade(sr);

                shaded_clr += select( hit_rec.hit, C(clr, S(1.0)), C(0.0) );
            }

            return shaded_clr;
        }
        else
        {
            return C(params.bg_color);
        }
    }
};

} // simple
} // visionaray

#endif // VSNRAY_SIMPLE_INL
