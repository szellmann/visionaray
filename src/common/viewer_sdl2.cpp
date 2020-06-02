// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdio>
#include <stdexcept>
#include <string>

#include <GL/glew.h>

#include <SDL2/SDL.h>

#include <imgui.h>

#include "input/key_event.h"
#include "input/keyboard.h"
#include "input/mouse.h"
#include "input/mouse_event.h"
#include "input/sdl2.h"
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
    // Set up new imgui frame
    ImGuiIO& io = ImGui::GetIO();
    IM_ASSERT(io.Fonts->IsBuilt() && "ImGui font atlas not built!");

    int w;
    int h;
    SDL_GetWindowSize(window, &w, &h);

    int display_w;
    int display_h;
    SDL_GL_GetDrawableSize(window, &display_w, &display_h);
    io.DisplaySize = ImVec2(static_cast<float>(w), static_cast<float>(h));

    if (w > 0 && h > 0)
    {
        io.DisplayFramebufferScale = ImVec2(
            static_cast<float>(display_w / w),
            static_cast<float>(display_h / h)
            );
    }
    ImGui::NewFrame();

    // Render
    viewer->on_display();

    // Draw imgui
    ImGui::Render();
    viewer->imgui_draw_opengl2(ImGui::GetDrawData());

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
    // imgui
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2(static_cast<float>(event.x), static_cast<float>(event.y));

    if (io.WantCaptureMouse)
    {
        return;
    }

    // viewer
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
    // imgui
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2(static_cast<float>(event.x), static_cast<float>(event.y));
    int imgui_button = event.button == SDL_BUTTON_LEFT ? 0 : event.button == SDL_BUTTON_RIGHT ? 1 : 2;
    io.MouseDown[imgui_button] = true;

    // viewer
    mouse::pos p = { event.x, event.y };

    auto b = mouse::map_sdl2_button(event.button);
    auto m = keyboard::map_sdl2_modifiers(SDL_GetModState());

    viewer->on_mouse_down( mouse_event(mouse::ButtonDown, p, b, m) );
    down_button = b;
}

void viewer_sdl2::impl::call_mouse_up(SDL_MouseButtonEvent const& event)
{
    // imgui
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2(static_cast<float>(event.x), static_cast<float>(event.y));
    int imgui_button = event.button == SDL_BUTTON_LEFT ? 0 : event.button == SDL_BUTTON_RIGHT ? 1 : 2;
    io.MouseDown[imgui_button] = false;

    // viewer
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

viewer_sdl2::viewer_sdl2(int width, int height, char const* window_title, display_mode_t display_mode)
    : viewer_base(width, height, window_title, display_mode)
    , impl_(new impl)
{
    impl_->viewer = this;
}

viewer_sdl2::~viewer_sdl2()
{
    imgui_destroy_font_texture_opengl2();

    ImGui::DestroyContext();

    SDL_GL_DeleteContext(impl_->context);
    SDL_DestroyWindow(impl_->window);
}

void viewer_sdl2::init(int argc, char** argv)
{
    viewer_base::init(argc, argv);


    // SDL window

    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        printf("Could not initialize initialize SDL: %s\n", SDL_GetError());
        return;
    }

    impl_->window = SDL_CreateWindow(
            window_title(),
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            width(),
            height(),
            SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL// | SDL_WINDOW_ALLOW_HIGHDPI
            );

    if (impl_->window == nullptr)
    {
        printf("Could not create SDL window: %s\n", SDL_GetError());
        return;
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

    GLenum error = glewInit();
    if (error != GLEW_OK)
    {
        std::string error_string("glewInit() failed: ");
        error_string.append(reinterpret_cast<char const*>(glewGetErrorString(error)));
        throw std::runtime_error(error_string);
    }


    // ImGui

    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    imgui_create_font_texture_opengl2();

    ImGuiIO& io = ImGui::GetIO();

    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
    io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;
    io.BackendPlatformName = "imgui_impl_sdl";

    io.KeyMap[ImGuiKey_Tab] = SDL_SCANCODE_TAB;
    io.KeyMap[ImGuiKey_LeftArrow] = SDL_SCANCODE_LEFT;
    io.KeyMap[ImGuiKey_RightArrow] = SDL_SCANCODE_RIGHT;
    io.KeyMap[ImGuiKey_UpArrow] = SDL_SCANCODE_UP;
    io.KeyMap[ImGuiKey_DownArrow] = SDL_SCANCODE_DOWN;
    io.KeyMap[ImGuiKey_PageUp] = SDL_SCANCODE_PAGEUP;
    io.KeyMap[ImGuiKey_PageDown] = SDL_SCANCODE_PAGEDOWN;
    io.KeyMap[ImGuiKey_Home] = SDL_SCANCODE_HOME;
    io.KeyMap[ImGuiKey_End] = SDL_SCANCODE_END;
    io.KeyMap[ImGuiKey_Insert] = SDL_SCANCODE_INSERT;
    io.KeyMap[ImGuiKey_Delete] = SDL_SCANCODE_DELETE;
    io.KeyMap[ImGuiKey_Backspace] = SDL_SCANCODE_BACKSPACE;
    io.KeyMap[ImGuiKey_Space] = SDL_SCANCODE_SPACE;
    io.KeyMap[ImGuiKey_Enter] = SDL_SCANCODE_RETURN;
    io.KeyMap[ImGuiKey_Escape] = SDL_SCANCODE_ESCAPE;
    io.KeyMap[ImGuiKey_KeyPadEnter] = SDL_SCANCODE_RETURN2;
    io.KeyMap[ImGuiKey_A] = SDL_SCANCODE_A;
    io.KeyMap[ImGuiKey_C] = SDL_SCANCODE_C;
    io.KeyMap[ImGuiKey_V] = SDL_SCANCODE_V;
    io.KeyMap[ImGuiKey_X] = SDL_SCANCODE_X;
    io.KeyMap[ImGuiKey_Y] = SDL_SCANCODE_Y;
    io.KeyMap[ImGuiKey_Z] = SDL_SCANCODE_Z;
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

void viewer_sdl2::resize(int width, int height)
{
    viewer_base::resize(width, height);
}

void viewer_sdl2::swap_buffers()
{
    SDL_GL_SwapWindow(impl_->window);
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

bool viewer_sdl2::have_imgui_support()
{
    return true;
}


//-------------------------------------------------------------------------------------------------
// Event handlers
//

void viewer_sdl2::on_idle()
{
    impl_->call_display();
}
