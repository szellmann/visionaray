// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_VIEWER_COCOA_H
#define VSNRAY_COMMON_VIEWER_COCOA_H 1

#include <memory>

#include "viewer_base.h"

namespace visionaray
{

class viewer_cocoa : public viewer_base
{
public:

    viewer_cocoa(
            int width,
            int height,
            char const* window_title    = "Visionaray Cocoa Viewer",
            display_mode_t display_mode = {}
            );
    virtual ~viewer_cocoa();

    void init(int argc, char** argv);

    void event_loop();
    void resize(int width, int height);
    void swap_buffers();
    void toggle_full_screen();
    void quit();

public:

    void call_on_display();
    void call_on_key_press(key_event const& event);
    void call_on_mouse_down(mouse_event const& event);
    void call_on_mouse_move(mouse_event const& event);
    void call_on_mouse_up(mouse_event const& event);
    void call_on_resize(int width, int height);

private:

    struct impl;
    std::unique_ptr<impl> const impl_;

};

} // visionaray

#endif // VSNRAY_COMMON_VIEWER_COCOA_H
