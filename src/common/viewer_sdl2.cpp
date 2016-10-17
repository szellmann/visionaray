// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <stdexcept>

#include <GL/glew.h>

#include <SDL2/SDL.h>

#include "input/keyboard.h"
#include "input/mouse.h"
#include "viewer_sdl2.h"

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Private implementation
//

struct viewer_sdl2::impl
{
    viewer_sdl2* viewer         = nullptr;
    SDL_Window* window          = nullptr;
    SDL_GLContext context;
    mouse::button down_button   = mouse::NoButton;

    void call_close();
    void call_display();
    void call_key_press(SDL_KeyboardEvent const& event);
    void call_key_release(SDL_KeyboardEvent const& event);
    void call_mouse_move(SDL_MouseMotionEvent const& event);
    void call_mouse_down(SDL_MouseButtonEvent const& event);
    void call_mouse_up(SDL_MouseButtonEvent const& event);
    void call_resize(int w, int h);
};


void viewer_sdl2::impl::call_close()
{
    viewer->on_close();
}

void viewer_sdl2::impl::call_display()
{
    viewer->on_display();
    SDL_GL_SwapWindow(window);
}

void viewer_sdl2::impl::call_key_press(SDL_KeyboardEvent const& event)
{
    auto k = keyboard::map_sdl2_key(event.keysym.sym, event.keysym.mod);
    auto m = keyboard::map_sdl2_modifiers(event.keysym.mod);

    viewer->on_key_press( key_event(keyboard::KeyPress, k, m) );
}

void viewer_sdl2::impl::call_key_release(SDL_KeyboardEvent const& event)
{
    auto k = keyboard::map_sdl2_key(event.keysym.sym);

    viewer->on_key_release( key_event(keyboard::KeyRelease, k) );
}

void viewer_sdl2::impl::call_mouse_move(SDL_MouseMotionEvent const& event)
{
    mouse::pos p = { event.x, event.y };

    mouse_event ev(
            mouse::Move,
            p,
            down_button,
            keyboard::NoKey
            );

    viewer->on_mouse_move(ev);
}

void viewer_sdl2::impl::call_mouse_down(SDL_MouseButtonEvent const& event)
{
    mouse::pos p = { event.x, event.y };

    auto b = mouse::map_sdl2_button(event.button);
    auto m = keyboard::map_sdl2_modifiers(SDL_GetModState());

    viewer->on_mouse_down( mouse_event(mouse::ButtonDown, p, b, m) );
    down_button = b;
}

void viewer_sdl2::impl::call_mouse_up(SDL_MouseButtonEvent const& event)
{
    mouse::pos p = { event.x, event.y };

    auto b = mouse::map_sdl2_button(event.button);
    auto m = keyboard::map_sdl2_modifiers(SDL_GetModState());

    viewer->on_mouse_up( mouse_event(mouse::ButtonUp, p, b, m) );
    down_button = mouse::NoButton;
}

void viewer_sdl2::impl::call_resize(int w, int h)
{
    viewer->on_resize(w, h);
}


//-------------------------------------------------------------------------------------------------
// viewer_sdl2
//

viewer_sdl2::viewer_sdl2(int width, int height, std::string window_title)
    : viewer_base(width, height, window_title)
    , impl_(new impl)
{
    impl_->viewer = this;
}

viewer_sdl2::~viewer_sdl2()
{
}

void viewer_sdl2::init(int argc, char** argv)
{
    viewer_base::init(argc, argv);


    // SDL window

    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        printf("Oups\n");
    }

    impl_->window = SDL_CreateWindow(
            window_title().c_str(),
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            width(),
            height(),
            SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL// | SDL_WINDOW_ALLOW_HIGHDPI
            );

    if (impl_->window == nullptr)
    {
        printf("Oups\n");
    }


    // Full screen mode

    if (full_screen())
    {
        SDL_SetWindowFullscreen(impl_->window, SDL_WINDOW_FULLSCREEN);
    }


    // OpenGL context

    impl_->context = SDL_GL_CreateContext(impl_->window);

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);


    // GLEW

    if (glewInit() != GLEW_OK)
    {
        throw std::runtime_error("glewInit() failed");
    }
}

void viewer_sdl2::event_loop()
{
    impl_->call_resize(width(), height());

    for (;;)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_QUIT:
                impl_->call_close();
                return;

            case SDL_KEYDOWN:
                impl_->call_key_press(event.key);
                break;

            case SDL_KEYUP:
                impl_->call_key_release(event.key);
                break;

            case SDL_MOUSEMOTION:
                impl_->call_mouse_move(event.motion);
                break;

            case SDL_MOUSEBUTTONDOWN:
                impl_->call_mouse_down(event.button);
                break;

            case SDL_MOUSEBUTTONUP:
                impl_->call_mouse_up(event.button);
                break;

            // Window events
            case SDL_WINDOWEVENT:
            {
                switch (event.window.event)
                {
                case SDL_WINDOWEVENT_EXPOSED:
                    impl_->call_display();
                    break;

                case SDL_WINDOWEVENT_RESIZED:
                case SDL_WINDOWEVENT_SIZE_CHANGED:
                    impl_->call_resize(event.window.data1, event.window.data2);
                    impl_->call_display();
                    break;

                case SDL_WINDOWEVENT_SHOWN:
                    impl_->call_resize(width(), height());
                    impl_->call_display();
                    break;

                default:
                    break;
                }

                break;
            }

            default:
                break;
            }
        }

        on_idle();
    }
}

void viewer_sdl2::toggle_full_screen()
{
    if (full_screen())
    {
        SDL_SetWindowFullscreen(impl_->window, 0);
    }
    else
    {
        SDL_SetWindowFullscreen(impl_->window, SDL_WINDOW_FULLSCREEN);
    }

    viewer_base::toggle_full_screen();
}

void viewer_sdl2::quit()
{
    SDL_Event quit;
    quit.type = SDL_QUIT;
    SDL_PushEvent(&quit);

    viewer_base::quit();
}

void viewer_sdl2::resize(int width, int height)
{
    viewer_base::resize(width, height);
}


//-------------------------------------------------------------------------------------------------
// Event handlers
//

void viewer_sdl2::on_idle()
{
    impl_->call_display();
}
