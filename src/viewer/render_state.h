// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VIEWER_STATE_H
#define VSNRAY_VIEWER_STATE_H 1

namespace visionaray
{

struct render_state
{
    enum mode_type
    {
        CPU,
        GPU
    };

    enum color_space_type
    {
        RGB,
        SRGB
    };

    mode_type mode = CPU;
    color_space_type color_space = RGB;
    bool direct_rendering = true;
};

} // visionaray

#endif // VSNRAY_VIEWER_STATE_H
