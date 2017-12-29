// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_KEY_EVENT_H
#define VSNRAY_COMMON_INPUT_KEY_EVENT_H 1

#include "keyboard.h"

namespace visionaray
{

class key_event
{
public:

    key_event(
            keyboard::event_type type,
            keyboard::key key
            )
        : type_(type)
        , key_(key)
        , modifiers_(keyboard::NoKey)
    {
    }

    key_event(
            keyboard::event_type type,
            keyboard::key key,
            keyboard::key_modifiers modifiers)
        : type_(type)
        , key_(key)
        , modifiers_(modifiers)
    {
    }

    keyboard::event_type type()         const { return type_; }
    keyboard::key key()                 const { return key_; }
    keyboard::key_modifiers modifiers() const { return modifiers_; }

private:

    keyboard::event_type type_;
    keyboard::key key_;
    keyboard::key_modifiers modifiers_;

};

} // visionaray

#endif // VSNRAY_COMMON_INPUT_KEY_EVENT_H
