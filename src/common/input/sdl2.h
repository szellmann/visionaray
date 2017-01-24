// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_SDL2_H
#define VSNRAY_COMMON_INPUT_SDL2_H 1

#include <common/config.h>

#if VSNRAY_HAVE_SDL2

#include <SDL2/SDL.h>

#include "keyboard.h"
#include "mouse.h"


namespace visionaray
{

namespace mouse
{

//-------------------------------------------------------------------------------------------------
// Mouse buttons
//

static inline buttons map_sdl2_button(Uint8 but)
{
    // TODO: multiple buttons
    switch (but)
    {

    case SDL_BUTTON_LEFT:
        return mouse::Left;
    case SDL_BUTTON_MIDDLE:
        return mouse::Middle;
    case SDL_BUTTON_RIGHT:
        return mouse::Right;
    default:
        return NoButton;

    }

    return NoButton;
}

} // mouse


namespace keyboard
{

//-------------------------------------------------------------------------------------------------
// Keys
//

static inline key map_sdl2_key(SDL_Keycode code, Uint16 modifiers = KMOD_NONE)
{
    bool shift = (modifiers & KMOD_LSHIFT) || (modifiers & KMOD_RSHIFT);

    switch (code)
    {

    case SDLK_a:        return shift ? A : a;
    case SDLK_b:        return shift ? B : b;
    case SDLK_c:        return shift ? C : c;
    case SDLK_d:        return shift ? D : d;
    case SDLK_e:        return shift ? E : e;
    case SDLK_f:        return shift ? F : f;
    case SDLK_g:        return shift ? G : g;
    case SDLK_h:        return shift ? H : h;
    case SDLK_i:        return shift ? I : i;
    case SDLK_j:        return shift ? J : j;
    case SDLK_k:        return shift ? K : k;
    case SDLK_l:        return shift ? L : l;
    case SDLK_m:        return shift ? M : m;
    case SDLK_n:        return shift ? N : n;
    case SDLK_o:        return shift ? O : o;
    case SDLK_p:        return shift ? P : p;
    case SDLK_q:        return shift ? Q : q;
    case SDLK_r:        return shift ? R : r;
    case SDLK_s:        return shift ? S : s;
    case SDLK_t:        return shift ? T : t;
    case SDLK_u:        return shift ? U : u;
    case SDLK_v:        return shift ? V : v;
    case SDLK_w:        return shift ? W : w;
    case SDLK_x:        return shift ? X : x;
    case SDLK_y:        return shift ? Y : y;
    case SDLK_z:        return shift ? Z : z;

    case SDLK_0:        return Zero;
    case SDLK_1:        return One;
    case SDLK_2:        return Two;
    case SDLK_3:        return Three;
    case SDLK_4:        return Four;
    case SDLK_5:        return Five;
    case SDLK_6:        return Six;
    case SDLK_7:        return Seven;
    case SDLK_8:        return Eight;
    case SDLK_9:        return Nine;

    case SDLK_SPACE:    return Space;
    case SDLK_ESCAPE:   return Escape;
    case SDLK_RETURN:   return Enter;
    case SDLK_TAB:      return Tab;

    case SDLK_F1:       return F1;
    case SDLK_F2:       return F2;
    case SDLK_F3:       return F3;
    case SDLK_F4:       return F4;
    case SDLK_F5:       return F5;
    case SDLK_F6:       return F6;
    case SDLK_F7:       return F7;
    case SDLK_F8:       return F8;
    case SDLK_F9:       return F9;
    case SDLK_F10:      return F10;
    case SDLK_F11:      return F11;
    case SDLK_F12:      return F12;

    default:            return NoKey;

    }

    return NoKey;
}


//-------------------------------------------------------------------------------------------------
// Modifiers
//

static inline key_modifiers map_sdl2_modifiers(Uint16 code)
{
    key_modifiers result = NoKey;

    if (code & KMOD_ALT)
    {
        result |= Alt;
    }

    if ((code & KMOD_LSHIFT) || (code & KMOD_RSHIFT))
    {
        result |= Shift;
    }

    return result;
}

} // keyboard
} // visionaray

#endif // VSNRAY_COMMON_INPUT_SDL2_H
