// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_KEYBOARD_H
#define VSNRAY_COMMON_INPUT_KEYBOARD_H 1

#include <common/config.h>

#include <cassert>
#include <bitset>

#include <visionaray/detail/platform.h>

#if VSNRAY_HAVE_GLUT

#if defined(VSNRAY_OS_DARWIN)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#endif // VSNRAY_HAVE_GLUT


#if VSNRAY_HAVE_QT5CORE
#include <Qt>
#endif // VSNRAY_HAVE_QT5CORE


#if VSNRAY_HAVE_SDL2
#include <SDL2/SDL.h>
#endif // VSNRAY_HAVE_SDL2


namespace visionaray
{
namespace keyboard
{

enum key
{
    A = 0x41, B = 0x42, C = 0x43, D = 0x44, E = 0x45,
    F = 0x46, G = 0x47, H = 0x48, I = 0x49, J = 0x4A,
    K = 0x4B, L = 0x4C, M = 0x4D, N = 0x4E, O = 0x4F,
    P = 0x50, Q = 0x51, R = 0x52, S = 0x53, T = 0x54,
    U = 0x55, V = 0x56, W = 0x57, X = 0x58, Y = 0x59,
    Z = 0x5A,

    a = 0x61, b = 0x62, c = 0x63, d = 0x64, e = 0x65,
    f = 0x66, g = 0x67, h = 0x68, i = 0x69, j = 0x6A,
    k = 0x6B, l = 0x6C, m = 0x6D, n = 0x6E, o = 0x6F,
    p = 0x70, q = 0x71, r = 0x72, s = 0x73, t = 0x74,
    u = 0x75, v = 0x76, w = 0x77, x = 0x78, y = 0x79,
    z = 0x7A,

    Zero = 0x30, One = 0x31, Two = 0x32, Three = 0x33, Four = 0x34,
    Five = 0x35, Six = 0x36, Seven = 0x37, Eight = 0x38, Nine = 0x39,

    Plus = 0x2B, Comma = 0x2C, Minus = 0x2D, Period = 0x2E,

    ArrowLeft, ArrowRight, ArrowUp, ArrowDown,

    PageUp, PageDown, Home, End, Insert,

    Space = 0x20, Escape = 0x1B, Enter = 0x0D, Tab = 0x9,

    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,

    // modifiers, can be combined bitwise

    Ctrl  = 0x00000001,
    Alt   = 0x00000002,
    Shift = 0x00000004,

    NoKey = 0x00000000
};


using key_modifiers = key;


enum event_type
{
    KeyPress = 0,
    KeyRelease
};


//-------------------------------------------------------------------------------------------------
// Bitwise operations on keys
//

inline key_modifiers operator&(key_modifiers a, key b)
{
    return static_cast<key_modifiers>( static_cast<int>(a) & static_cast<int>(b) );
}

inline key_modifiers operator|(key_modifiers a, key b)
{
    return static_cast<key_modifiers>( static_cast<int>(a) | static_cast<int>(b) );
}

inline key_modifiers operator^(key_modifiers a, key b)
{
    return static_cast<key_modifiers>( static_cast<int>(a) ^ static_cast<int>(b) );
}

inline key_modifiers& operator&=(key_modifiers& a, key b)
{
    a = a & b;
    return a;
}

inline key_modifiers& operator|=(key_modifiers& a, key b)
{
    a = a | b;
    return a;
}

inline key_modifiers& operator^=(key_modifiers& a, key b)
{
    a = a ^ b;
    return a;
}


#if VSNRAY_HAVE_GLUT

//-------------------------------------------------------------------------------------------------
// Map GLUT entities
//

static inline key map_glut_key(unsigned char code)
{

    switch (code)
    {

    case 'A':                   return A;
    case 'B':                   return B;
    case 'C':                   return C;
    case 'D':                   return D;
    case 'E':                   return E;
    case 'F':                   return F;
    case 'G':                   return G;
    case 'H':                   return H;
    case 'I':                   return I;
    case 'J':                   return J;
    case 'K':                   return K;
    case 'L':                   return L;
    case 'M':                   return M;
    case 'N':                   return N;
    case 'O':                   return O;
    case 'P':                   return P;
    case 'Q':                   return Q;
    case 'R':                   return R;
    case 'S':                   return S;
    case 'T':                   return T;
    case 'U':                   return U;
    case 'V':                   return V;
    case 'W':                   return W;
    case 'X':                   return X;
    case 'Y':                   return Y;
    case 'Z':                   return Z;

    case 'a':                   return a;
    case 'b':                   return b;
    case 'c':                   return c;
    case 'd':                   return d;
    case 'e':                   return e;
    case 'f':                   return f;
    case 'g':                   return g;
    case 'h':                   return h;
    case 'i':                   return i;
    case 'j':                   return j;
    case 'k':                   return k;
    case 'l':                   return l;
    case 'm':                   return m;
    case 'n':                   return n;
    case 'o':                   return o;
    case 'p':                   return p;
    case 'q':                   return q;
    case 'r':                   return r;
    case 's':                   return s;
    case 't':                   return t;
    case 'u':                   return u;
    case 'v':                   return v;
    case 'w':                   return w;
    case 'x':                   return x;
    case 'y':                   return y;
    case 'z':                   return z;


    case '0':                   return Zero;
    case '1':                   return One;
    case '2':                   return Two;
    case '3':                   return Three;
    case '4':                   return Four;
    case '5':                   return Five;
    case '6':                   return Six;
    case '7':                   return Seven;
    case '8':                   return Eight;
    case '9':                   return Nine;

    case '+':                   return Plus;
    case ',':                   return Comma;
    case '-':                   return Minus;
    case '.':                   return Period;

    case 0x20:                  return Space;
    case 0x1B:                  return Escape;
    case 0x0D:                  return Enter;
    case 0x9:                   return Tab;

    }

    return NoKey;
}


static inline key map_glut_special(int code)
{
    switch (code)
    {

    case GLUT_KEY_LEFT:         return ArrowLeft;
    case GLUT_KEY_RIGHT:        return ArrowRight;
    case GLUT_KEY_UP:           return ArrowUp;
    case GLUT_KEY_DOWN:         return ArrowDown;

    case GLUT_KEY_PAGE_UP:      return PageUp;
    case GLUT_KEY_PAGE_DOWN:    return PageDown;
    case GLUT_KEY_HOME:         return Home;
    case GLUT_KEY_END:          return End;
    case GLUT_KEY_INSERT:       return Insert;

    case GLUT_KEY_F1:           return F1;
    case GLUT_KEY_F2:           return F2;
    case GLUT_KEY_F3:           return F3;
    case GLUT_KEY_F4:           return F4;
    case GLUT_KEY_F5:           return F5;
    case GLUT_KEY_F6:           return F6;
    case GLUT_KEY_F7:           return F7;
    case GLUT_KEY_F8:           return F8;
    case GLUT_KEY_F9:           return F9;
    case GLUT_KEY_F10:          return F10;
    case GLUT_KEY_F11:          return F11;
    case GLUT_KEY_F12:          return F12;

    }


    return NoKey;
}


static inline key_modifiers map_glut_modifiers(unsigned char code)
{
    key_modifiers result = NoKey;

    if (code & GLUT_ACTIVE_CTRL)
    {
        result |= Ctrl;
    }

    if (code & GLUT_ACTIVE_ALT)
    {
        result |= Alt;
    }

    if (code & GLUT_ACTIVE_SHIFT)
    {
        result |= Shift;
    }

    return result;
}

#endif // VSNRAY_HAVE_GLUT


#if VSNRAY_HAVE_QT5CORE

static inline key map_qt_key(int code, Qt::KeyboardModifiers modifiers = Qt::NoModifier)
{
    bool shift = modifiers & Qt::ShiftModifier;

    switch (code)
    {

    case Qt::Key_A:         return shift ? A : a;
    case Qt::Key_B:         return shift ? B : b;
    case Qt::Key_C:         return shift ? C : c;
    case Qt::Key_D:         return shift ? D : d;
    case Qt::Key_E:         return shift ? E : e;
    case Qt::Key_F:         return shift ? F : f;
    case Qt::Key_G:         return shift ? G : g;
    case Qt::Key_H:         return shift ? H : h;
    case Qt::Key_I:         return shift ? I : i;
    case Qt::Key_J:         return shift ? J : j;
    case Qt::Key_K:         return shift ? K : k;
    case Qt::Key_L:         return shift ? L : l;
    case Qt::Key_M:         return shift ? M : m;
    case Qt::Key_N:         return shift ? N : n;
    case Qt::Key_O:         return shift ? O : o;
    case Qt::Key_P:         return shift ? P : p;
    case Qt::Key_Q:         return shift ? Q : q;
    case Qt::Key_R:         return shift ? R : r;
    case Qt::Key_S:         return shift ? S : s;
    case Qt::Key_T:         return shift ? T : t;
    case Qt::Key_U:         return shift ? U : u;
    case Qt::Key_V:         return shift ? V : v;
    case Qt::Key_W:         return shift ? W : w;
    case Qt::Key_X:         return shift ? X : x;
    case Qt::Key_Y:         return shift ? Y : y;
    case Qt::Key_Z:         return shift ? Z : z;

    case Qt::Key_0:         return Zero;
    case Qt::Key_1:         return One;
    case Qt::Key_2:         return Two;
    case Qt::Key_3:         return Three;
    case Qt::Key_4:         return Four;
    case Qt::Key_5:         return Five;
    case Qt::Key_6:         return Six;
    case Qt::Key_7:         return Seven;
    case Qt::Key_8:         return Eight;
    case Qt::Key_9:         return Nine;

    case Qt::Key_Plus:      return Plus;
    case Qt::Key_Comma:     return Comma;
    case Qt::Key_Minus:     return Minus;
    case Qt::Key_Period:    return Period;

    case Qt::Key_Space:     return Space;
    case Qt::Key_Escape:    return Escape;
    case Qt::Key_Enter:     return Enter;
    case Qt::Key_Tab:       return Tab;

    case Qt::Key_Left:      return ArrowLeft;
    case Qt::Key_Right:     return ArrowRight;
    case Qt::Key_Up:        return ArrowUp;
    case Qt::Key_Down:      return ArrowDown;

    case Qt::Key_PageUp:    return PageUp;
    case Qt::Key_PageDown:  return PageDown;
    case Qt::Key_Home:      return Home;
    case Qt::Key_End:       return End;
    case Qt::Key_Insert:    return Insert;

    case Qt::Key_F1:        return F1;
    case Qt::Key_F2:        return F2;
    case Qt::Key_F3:        return F3;
    case Qt::Key_F4:        return F4;
    case Qt::Key_F5:        return F5;
    case Qt::Key_F6:        return F6;
    case Qt::Key_F7:        return F7;
    case Qt::Key_F8:        return F8;
    case Qt::Key_F9:        return F9;
    case Qt::Key_F10:       return F10;
    case Qt::Key_F11:       return F11;
    case Qt::Key_F12:       return F12;

    default:                return NoKey;

    }

    return NoKey;
}

static inline key_modifiers map_qt_modifiers(Qt::KeyboardModifiers code)
{
    key_modifiers result = NoKey;

#if defined(VSNRAY_OS_DARWIN)
    if (code & Qt::MetaModifier)
    {
        result |= Ctrl;
    }
#else
    if (code & Qt::ControlModifier)
    {
        result |= Ctrl;
    }
#endif

    if (code & Qt::AltModifier)
    {
        result |= Alt;
    }

    if (code & Qt::ShiftModifier)
    {
        result |= Shift;
    }

    return result;
}

#endif // VSNRAY_HAVE_QT5CORE


#if VSNRAY_HAVE_SDL2

//-------------------------------------------------------------------------------------------------
// Map SDL2 entities
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

#endif // VSNRAY_HAVE_SDL2

} // keyboard


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

    keyboard::event_type get_type()     const { return type_; }
    keyboard::key key()                 const { return key_; }
    keyboard::key_modifiers modifiers() const { return modifiers_; }

private:

    keyboard::event_type type_;
    keyboard::key key_;
    keyboard::key_modifiers modifiers_;

};

} // visionaray

#endif // VSNRAY_COMMON_INPUT_KEYBOARD_H
