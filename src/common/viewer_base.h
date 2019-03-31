// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_VIEWER_BASE_H
#define VSNRAY_COMMON_VIEWER_BASE_H 1

#include <memory>
#include <set>
#include <string>

#include <visionaray/math/vector.h>

namespace support
{
namespace cl
{
class CmdLine;
class OptionBase;
} // cl
} // support


namespace visionaray
{

class camera_manipulator;
class key_event;
class mouse_event;
class space_mouse_event;

class viewer_base
{
public:

    viewer_base(
            int width                   = 512,
            int height                  = 512,
            char const* window_title    = ""
            );
    virtual ~viewer_base();

    void init(int argc, char** argv);

    void parse_inifile(std::set<std::string> const& filenames);

    void add_manipulator( std::shared_ptr<camera_manipulator> manip );
    void add_cmdline_option( std::shared_ptr<support::cl::OptionBase> option );

    char const* window_title() const;
    bool full_screen() const;
    int width() const;
    int height() const;
    vec3 background_color() const;

    // Allow for unknown or unhandled command line arguments
    void set_allow_unknown_cmd_line_args(bool allow);

    // Returns a reference to the command line instance
    support::cl::CmdLine& cmd_line_inst();

    void set_background_color(vec3 color);

    virtual void event_loop();
    virtual void resize(int width, int height);
    virtual void swap_buffers();
    virtual void toggle_full_screen();
    virtual void quit();

    // Default: false. Reimplement if derived viewer has ImGui support
    static bool have_imgui_support();

protected:

    virtual void on_close();
    virtual void on_display();
    virtual void on_idle();
    virtual void on_key_press(key_event const& event);
    virtual void on_key_release(key_event const& event);
    virtual void on_mouse_move(mouse_event const& event);
    virtual void on_mouse_down(mouse_event const& event);
    virtual void on_mouse_up(mouse_event const& event);
    virtual void on_space_mouse_move(space_mouse_event const& event);
    virtual void on_resize(int w, int h);

private:

    struct impl;
    std::unique_ptr<impl> const impl_;

};

} // visionaray

#endif // VSNRAY_COMMON_VIEWER_BASE_H
