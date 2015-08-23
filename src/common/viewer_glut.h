// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_COMMON_VIEWER_GLUT_H
#define VSNRAY_COMMON_VIEWER_GLUT_H 1

#include <memory>
#include <string>

#include "viewer_base.h"

namespace visionaray
{

class camera_manipulator;
class key_event;
class mouse_event;

class viewer_glut : public viewer_base
{
public:

    viewer_glut(
            int width                   = 512,
            int height                  = 512,
            std::string window_title    = "Visionaray GLUT Viewer"
            );
    virtual ~viewer_glut();

    void init(int argc, char** argv);

    void event_loop();
    void swap_buffers();
    void resize(int width, int height);
    void toggle_full_screen();

protected:

    virtual void on_idle();

private:

    struct impl;
    friend struct impl;
    std::unique_ptr<impl> impl_;

};

} // visionaray

#endif // VSNRAY_COMMON_VIEWER_GLUT_H
