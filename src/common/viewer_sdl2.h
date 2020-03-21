// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_VIEWER_SDL2_H
#define VSNRAY_COMMON_VIEWER_SDL2_H 1

#include <memory>

#include "viewer_base.h"

namespace visionaray
{

class viewer_sdl2 : public viewer_base
{
public:

    viewer_sdl2(
            int width,
            int height,
            char const* window_title    = "Visionaray SDL2 Viewer"
            );
    virtual ~viewer_sdl2();

    void init(int argc, char** argv);

    void event_loop();
    void resize(int width, int height);
    void swap_buffers();
    void toggle_full_screen();
    void quit();

    static bool have_imgui_support();

protected:

    virtual void on_idle();

private:

    struct impl;
    std::unique_ptr<impl> const impl_;

};

} // visionaray

#endif // VSNRAY_COMMON_VIEWER_SDL2_H
