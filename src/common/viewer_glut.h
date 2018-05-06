// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_VIEWER_GLUT_H
#define VSNRAY_COMMON_VIEWER_GLUT_H 1

#include <memory>

#include "viewer_base.h"

namespace visionaray
{

class viewer_glut : public viewer_base
{
public:

    viewer_glut(
            int width                   = 512,
            int height                  = 512,
            char const* window_title    = "Visionaray GLUT Viewer"
            );
    virtual ~viewer_glut();

    void init(int argc, char** argv);

    void event_loop();
    void resize(int width, int height);
    void swap_buffers();
    void toggle_full_screen();
    void quit();

protected:

    virtual void on_idle();

private:

    struct impl;
    friend struct impl;
    std::unique_ptr<impl> const impl_;

};

} // visionaray

#endif // VSNRAY_COMMON_VIEWER_GLUT_H
