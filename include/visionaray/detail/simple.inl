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

        typedef typename R::scalar_type scalar_type;

        C bg_color
        (
            scalar_type(0.1),
            scalar_type(0.4),
            scalar_type(1.0),
            scalar_type(1.0)
        );


        C shaded_clr(scalar_type(0.0));

        auto hit_rec = closest_hit(ray, params.prims.begin, params.prims.end);

        if (any(hit_rec.hit))
        {
            auto surf = get_surface(hit_rec, params);

            for (auto it = params.lights.begin; it != params.lights.end; ++it)
            {
                auto sr     = make_shade_record<Params, scalar_type>();
                sr.active   = hit_rec.hit;
                sr.view_dir = -ray.dir;
                sr.light    = it;
                auto clr    = surf.shade(sr);

                shaded_clr += select( hit_rec.hit, C(clr, scalar_type(1.0)), bg_color );
            }

            return shaded_clr;
        }
        else
        {
            return bg_color;
        }
    }
};

} // simple
} // visionaray

#endif // VSNRAY_SIMPLE_INL


