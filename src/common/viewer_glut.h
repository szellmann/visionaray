// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_COMMON_VIEWER_GLUT_H
#define VSNRAY_COMMON_VIEWER_GLUT_H

#include <memory>
#include <string>

namespace support
{
namespace cl
{
class OptionBase;
} // cl
} // support


namespace visionaray
{

class camera_manipulator;
class key_event;
class mouse_event;

class viewer_glut
{
public:

    viewer_glut(
            int width                   = 512,
            int height                  = 512,
            std::string window_title    = "Visionaray GLUT Viewer"
            );
    virtual ~viewer_glut();

    void init(int argc, char** argv);

    void add_manipulator( std::shared_ptr<camera_manipulator> manip );
    void add_cmdline_option( std::shared_ptr<support::cl::OptionBase> option );
    void event_loop();
    void swap_buffers();
    void toggle_full_screen();

    int width();
    int height();
    vec3 background_color() const;

protected:

    virtual void on_display();
    virtual void on_idle();
    virtual void on_key_press(key_event const& event);
    virtual void on_key_release(key_event const& event);
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
