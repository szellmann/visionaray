// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_SPACE_MOUSE_H
#define VSNRAY_COMMON_INPUT_SPACE_MOUSE_H 1

#include <cassert>

#include <visionaray/math/forward.h>
#include <visionaray/math/vector.h>

namespace visionaray
{

class space_mouse_event;

namespace space_mouse
{

enum event_type
{
    Button,
    Rotation,
    Translation,

    EventTypeCount // last
};

enum button
{
    Button1     = 0x00000001,
    Button2     = 0x00000002,
    Button3     = 0x00000004,
    Button4     = 0x00000008,
    Button5     = 0x00000010,
    Button6     = 0x00000020,
    Button7     = 0x00000040,
    Button8     = 0x00000080,

    Button9     = 0x00000100,
    Button10    = 0x00000200,
    Button11    = 0x00000400,
    Button12    = 0x00000800,
    Button13    = 0x00001000,
    Button14    = 0x00002000,
    Button15    = 0x00004000,
    Button16    = 0x00008000,

    Button17    = 0x00010000,
    Button18    = 0x00020000,
    Button19    = 0x00040000,
    Button20    = 0x00080000,
    Button21    = 0x00100000,
    Button22    = 0x00200000,
    Button23    = 0x00400000,
    Button24    = 0x00800000,

    Button25    = 0x01000000,
    Button26    = 0x02000000,
    Button27    = 0x04000000,
    Button28    = 0x08000000,
    Button29    = 0x10000000,
    Button30    = 0x20000000,
    Button31    = 0x40000000,
    Button32    = 0x80000000,

    ButtonCount = 0xFFFFFFFF // last
};

using pos = vec3i;

typedef void (*event_callback)(space_mouse_event const&);

bool init();
void register_event_callback(event_type type, event_callback cb);
void cleanup();

} // space_mouse

class space_mouse_event
{
public:

    // Axis changed
    space_mouse_event(
            space_mouse::event_type type,
            space_mouse::pos const& pos
            )
        : type_(type)
        , pos_(pos)
    {
        assert(type == space_mouse::Rotation || type == space_mouse::Translation);
    }

    // Buttons pressed
    space_mouse_event(space_mouse::event_type type, int buttons)
        : type_(type)
        , buttons_(buttons)
    {
        assert(type == space_mouse::Button);
    }

    space_mouse::event_type type() const { return type_; }
    space_mouse::pos const& pos()  const { return pos_; }
    int buttons()                  const { return buttons_; }

private:
    space_mouse::event_type type_;
    space_mouse::pos pos_;
    int buttons_;

};

} // visionaray

#endif // VSNRAY_COMMON_INPUT_SPACE_MOUSE_H
