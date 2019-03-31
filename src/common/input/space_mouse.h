// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_SPACE_MOUSE_H
#define VSNRAY_COMMON_INPUT_SPACE_MOUSE_H 1

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

using pos = vec3i;

typedef void (*event_callback)(space_mouse_event const&);

bool init();
void register_event_callback(event_type type, event_callback cb);
void cleanup();

} // space_mouse

class space_mouse_event
{
public:

    space_mouse_event(
            space_mouse::event_type type,
            space_mouse::pos const& pos
            )
        : type_(type)
        , pos_(pos)
    {
    }

    space_mouse::event_type type() const { return type_; }
    space_mouse::pos const& pos()  const { return pos_; }

private:
    space_mouse::event_type type_;
    space_mouse::pos pos_;

};

} // visionaray

#endif // VSNRAY_COMMON_INPUT_SPACE_MOUSE_H
