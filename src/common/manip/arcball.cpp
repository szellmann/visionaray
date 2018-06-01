// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "arcball.h"

using namespace visionaray;


arcball::arcball(float r)
    : radius(r)
{
}


//-------------------------------------------------------------------------------------------------
// Project x/y screen space position to ball coordinates
//

vec3 arcball::project(int x, int y, recti const& viewport)
{

    vec3 v(0.0f);

    x -= viewport.x;
    y -= viewport.y;

    auto width  = viewport.w;
    auto height = viewport.h;

#if 0

    // trackball

    v[0] =  (x - 0.5f * width ) / width;
    v[1] = -(y - 0.5f * height) / height;

    vec2 tmp(v[0], v[1]);
    float d = normh2(tmp);
    float r2 = radius * radius;

    if (d < radius * (1.0f / sqrt(2.0)))
    {
        v[2] = sqrt(r2 - d * d);
    }
    else
    {
        v[2] = r2 / (2.0f * d);
    }
#else

    // arcball

    v[0] =  (x - 0.5f * width ) / (radius * 0.5f * width );
    v[1] = -(y - 0.5f * height) / (radius * 0.5f * height);

    vec2 tmp(v[0], v[1]);
    float d = norm2(tmp);


    if (d > 1.0f)
    {
        float length = sqrt(d);

        v[0] /= length;
        v[1] /= length;
    }
    else
    {
        v[2] = sqrt(1.0f - d);
    }
#endif

    return v;

}
