// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_VIEWER_GLUT_H
#define VSNRAY_COMMON_VIEWER_GLUT_H 1

#include <memory>

#include "export.h"
#include "viewer_base.h"

namespace visionaray
{

class viewer_glut : public viewer_base
{
public:

    VSNRAY_COMMON_EXPORT viewer_glut(
            int width                   = 512,
            int height                  = 512,
            char const* window_title    = "Visionaray GLUT Viewer",
            display_mode_t display_mode = {}
            );
    VSNRAY_COMMON_EXPORT virtual ~viewer_glut();

    VSNRAY_COMMON_EXPORT void init(int argc, char** argv);

    VSNRAY_COMMON_EXPORT void event_loop();
    VSNRAY_COMMON_EXPORT void resize(int width, int height);
    VSNRAY_COMMON_EXPORT void swap_buffers();
    VSNRAY_COMMON_EXPORT void toggle_full_screen();
    VSNRAY_COMMON_EXPORT void quit();

    VSNRAY_COMMON_EXPORT static bool have_imgui_support();

protected:

    VSNRAY_COMMON_EXPORT virtual void on_idle();

private:

    struct impl;
    friend struct impl;
    std::unique_ptr<impl> const impl_;

};

} // visionaray

#endif // VSNRAY_COMMON_VIEWER_GLUT_H
