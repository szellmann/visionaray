// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_COMMON_VIEWER_GLUT_H
#define VSNRAY_COMMON_VIEWER_GLUT_H

#include <memory>
#include <string>

namespace visionaray
{

class camera_manipulator;
class mouse_event;

class viewer_glut
{
public:

    viewer_glut(int argc, char** argv);
    viewer_glut(std::string window_title, int argc, char** argv);
    viewer_glut(
            int width,
            int height,
            std::string window_title,
            int argc,
            char** argv
            );
    virtual ~viewer_glut();

    void add_manipulator( std::shared_ptr<camera_manipulator> manip );
    void event_loop();
    void swap_buffers();

    int width();
    int height();

protected:

    virtual void on_display();
    virtual void on_idle();
    virtual void on_key_press(unsigned char key); // TODO: keybaord event
    virtual void on_mouse_move(mouse_event const& event);
    virtual void on_mouse_down(mouse_event const& event);
    virtual void on_mouse_up(mouse_event const& event);
    virtual void on_resize(int w, int h);

private:

    struct impl;
    friend struct impl;
    std::unique_ptr<impl> impl_;

};

} // visionaray

#endif // VSNRAY_COMMON_VIEWER_GLUT_H
