// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_VIEWER_BASE_H
#define VSNRAY_COMMON_VIEWER_BASE_H 1

#include <memory>
#include <set>
#include <string>

#include <visionaray/math/forward.h>
#include <visionaray/math/vector.h>

#include "export.h"

namespace support
{
namespace cl
{
class CmdLine;
class OptionBase;
} // cl
} // support

struct ImDrawData;

namespace visionaray
{

class camera_manipulator;
class key_event;
class mouse_event;
class space_mouse_event;

struct display_mode_t
{
    bool double_buffered = true;
    bool multisampling   = false;
};

class viewer_base
{
public:

    VSNRAY_COMMON_EXPORT viewer_base(
            int width                   = 512,
            int height                  = 512,
            char const* window_title    = "",
            display_mode_t display_mode  = {}
            );
    VSNRAY_COMMON_EXPORT virtual ~viewer_base();

    VSNRAY_COMMON_EXPORT void init(int argc, char** argv);

    VSNRAY_COMMON_EXPORT void parse_inifile(std::set<std::string> const& filenames);

    VSNRAY_COMMON_EXPORT void add_manipulator( std::shared_ptr<camera_manipulator> manip );
    VSNRAY_COMMON_EXPORT void add_cmdline_option( std::shared_ptr<support::cl::OptionBase> option );

    VSNRAY_COMMON_EXPORT char const* window_title() const;
    VSNRAY_COMMON_EXPORT bool full_screen() const;
    VSNRAY_COMMON_EXPORT int width() const;
    VSNRAY_COMMON_EXPORT int height() const;
    VSNRAY_COMMON_EXPORT display_mode_t display_mode() const;
    VSNRAY_COMMON_EXPORT vec3 background_color() const;

    // Allow for unknown or unhandled command line arguments
    VSNRAY_COMMON_EXPORT void set_allow_unknown_cmd_line_args(bool allow);

    // Returns a reference to the command line instance
    VSNRAY_COMMON_EXPORT support::cl::CmdLine& cmd_line_inst();

    VSNRAY_COMMON_EXPORT void set_background_color(vec3 color);

    VSNRAY_COMMON_EXPORT virtual void set_window_title(char const* window_title);
    VSNRAY_COMMON_EXPORT virtual void event_loop();
    VSNRAY_COMMON_EXPORT virtual void resize(int width, int height);
    VSNRAY_COMMON_EXPORT virtual void swap_buffers();
    VSNRAY_COMMON_EXPORT virtual void toggle_full_screen();
    VSNRAY_COMMON_EXPORT virtual void quit();

    // Default: false. Reimplement if derived viewer has ImGui support
    VSNRAY_COMMON_EXPORT static bool have_imgui_support();

protected:

    VSNRAY_COMMON_EXPORT virtual void on_close();
    VSNRAY_COMMON_EXPORT virtual void on_display();
    VSNRAY_COMMON_EXPORT virtual void on_idle();
    VSNRAY_COMMON_EXPORT virtual void on_key_press(key_event const& event);
    VSNRAY_COMMON_EXPORT virtual void on_key_release(key_event const& event);
    VSNRAY_COMMON_EXPORT virtual void on_mouse_move(mouse_event const& event);
    VSNRAY_COMMON_EXPORT virtual void on_mouse_down(mouse_event const& event);
    VSNRAY_COMMON_EXPORT virtual void on_mouse_up(mouse_event const& event);
    VSNRAY_COMMON_EXPORT virtual void on_space_mouse_move(space_mouse_event const& event);
    VSNRAY_COMMON_EXPORT virtual void on_space_mouse_button_press(space_mouse_event const& event);
    VSNRAY_COMMON_EXPORT virtual void on_resize(int w, int h);

    void imgui_create_font_texture_opengl2();
    void imgui_destroy_font_texture_opengl2();
    void imgui_draw_opengl2(ImDrawData* draw_data);

private:

    struct impl;
    std::unique_ptr<impl> const impl_;

};

} // visionaray

#endif // VSNRAY_COMMON_VIEWER_BASE_H
