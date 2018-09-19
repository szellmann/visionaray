// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <GL/glew.h>

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN)

#include <AvailabilityMacros.h>

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

    #pragma GCC diagnostic ignored "-Wdeprecated"
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#endif

#include <OpenGL/gl.h>
#include <GLUT/glut.h>

#else // VSNRAY_OS_DARWIN

#if defined(VSNRAY_OS_WIN32)

#include <windows.h>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#endif
#include <GL/gl.h>
#include <GL/glut.h>
#ifdef FREEGLUT
#include <GL/freeglut_ext.h>
#endif

#endif

#include <imgui.h>

#include "input/glut.h"
#include "input/key_event.h"
#include "input/keyboard.h"
#include "input/mouse.h"
#include "input/mouse_event.h"
#include "viewer_glut.h"


using namespace visionaray;

struct viewer_glut::impl
{
    static viewer_glut*     viewer;
    static mouse::button    down_button;
    static int              win_id;
    static GLuint           imgui_font_texture;

    impl(viewer_glut* instance);

    void init(
            int argc,
            char** argv,
            char const* window_title,
            bool full_screen,
            int width,
            int height
            );
    void cleanup();

    static void close_func();
    static void display_func();
    static void idle_func();
    static void keyboard_func(unsigned char key, int, int);
    static void keyboard_up_func(unsigned char key, int, int);
    static void motion_func(int x, int y);
    static void mouse_func(int button, int state, int x, int y);
    static void passive_motion_func(int x, int y);
    static void reshape_func(int w, int h);
    static void special_func(int key, int, int);
    static void special_up_func(int key, int, int);

    static void imgui_create_font_texture();
    static void imgui_destroy_font_texture();
    static void imgui_draw(ImDrawData* draw_data);
};

viewer_glut*    viewer_glut::impl::viewer             = nullptr;
mouse::button   viewer_glut::impl::down_button        = mouse::NoButton;
int             viewer_glut::impl::win_id             = 0;
GLuint          viewer_glut::impl::imgui_font_texture = 0;


//-------------------------------------------------------------------------------------------------
// Private implementation methods
//-------------------------------------------------------------------------------------------------

viewer_glut::impl::impl(viewer_glut* instance)
{
    viewer_glut::impl::viewer = instance;
}


void viewer_glut::impl::imgui_create_font_texture()
{
    ImGuiIO& io = ImGui::GetIO();

    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    GLint prev_tex = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_tex);
    glGenTextures(1, &imgui_font_texture);
    glBindTexture(GL_TEXTURE_2D, imgui_font_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0); 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    io.Fonts->TexID = reinterpret_cast<ImTextureID>(static_cast<intptr_t>(imgui_font_texture));

    glBindTexture(GL_TEXTURE_2D, prev_tex);
}

void viewer_glut::impl::imgui_destroy_font_texture()
{
    assert(imgui_font_texture);

    ImGuiIO& io = ImGui::GetIO();

    glDeleteTextures(1, &imgui_font_texture);
    io.Fonts->TexID = nullptr;
    imgui_font_texture = 0;
}

void viewer_glut::impl::imgui_draw(ImDrawData* draw_data)
{
    ImGuiIO& io = ImGui::GetIO();

    int width = static_cast<int>(draw_data->DisplaySize.x * io.DisplayFramebufferScale.x);
    int height = static_cast<int>(draw_data->DisplaySize.y * io.DisplayFramebufferScale.y);

    if (width == 0 || height == 0)
    {
        return;
    }

    draw_data->ScaleClipRects(io.DisplayFramebufferScale);

    // Store OpenGL state
    GLint prev_texture = 0;
    GLint prev_polygon_mode[2] = {};
    GLint prev_viewport[4] = {};
    GLint prev_scissor_box[4] = {};
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_texture);
    glGetIntegerv(GL_POLYGON_MODE, prev_polygon_mode);
    glGetIntegerv(GL_VIEWPORT, prev_viewport);
    glGetIntegerv(GL_SCISSOR_BOX, prev_scissor_box); 

    glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_TRANSFORM_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glEnable(GL_SCISSOR_TEST);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnable(GL_TEXTURE_2D);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    //glUseProgram(0);

    glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(
        draw_data->DisplayPos.x,
        draw_data->DisplayPos.x + draw_data->DisplaySize.x,
        draw_data->DisplayPos.y + draw_data->DisplaySize.y,
        draw_data->DisplayPos.y,
        -1.0f,
        1.0f
        );
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    ImVec2 pos = draw_data->DisplayPos;
    for (int n = 0; n < draw_data->CmdListsCount; ++n)
    {
        ImDrawList const* cmd_list = draw_data->CmdLists[n];
        ImDrawVert const* vtx_buffer = cmd_list->VtxBuffer.Data;
        ImDrawIdx const* idx_buffer = cmd_list->IdxBuffer.Data;

        glVertexPointer(
            2,
            GL_FLOAT,
            sizeof(ImDrawVert),
            static_cast<GLvoid const*>(reinterpret_cast<char const*>(vtx_buffer) + IM_OFFSETOF(ImDrawVert, pos))
            );

        glTexCoordPointer(
            2,
            GL_FLOAT,
            sizeof(ImDrawVert),
            static_cast<GLvoid const*>(reinterpret_cast<char const*>(vtx_buffer) + IM_OFFSETOF(ImDrawVert, uv))
            );

        glColorPointer(
            4,
            GL_UNSIGNED_BYTE,
            sizeof(ImDrawVert),
            static_cast<GLvoid const*>(reinterpret_cast<char const*>(vtx_buffer) + IM_OFFSETOF(ImDrawVert, col))
            );

        for (int i = 0; i < cmd_list->CmdBuffer.Size; ++i)
        {
            ImDrawCmd const* pcmd = &cmd_list->CmdBuffer[i];

            if (pcmd->UserCallback)
            {
                pcmd->UserCallback(cmd_list, pcmd);
            }
            else
            {
                ImVec4 clip_rect(
                    pcmd->ClipRect.x - pos.x,
                    pcmd->ClipRect.y - pos.y,
                    pcmd->ClipRect.z - pos.x,
                    pcmd->ClipRect.w - pos.y
                    );

                if (clip_rect.x < width && clip_rect.y < height && clip_rect.z >= 0.0f && clip_rect.w >= 0.0f)
                {
                    glScissor(
                        static_cast<int>(clip_rect.x),
                        static_cast<int>(width - clip_rect.w),
                        static_cast<int>(clip_rect.z - clip_rect.x),
                        static_cast<int>(clip_rect.w - clip_rect.y)
                        );

                    glBindTexture(
                        GL_TEXTURE_2D,
                        static_cast<GLuint>(reinterpret_cast<intptr_t>(pcmd->TextureId))
                        );

                    glDrawElements(
                        GL_TRIANGLES,
                        static_cast<GLsizei>(pcmd->ElemCount),
                        sizeof(ImDrawIdx) == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT,
                        idx_buffer
                        );
                }
            }

            idx_buffer += pcmd->ElemCount;
        }
    }

    // Restore OpenGL state
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prev_texture));
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();
    glPolygonMode(GL_FRONT, static_cast<GLenum>(prev_polygon_mode[0]));
    glPolygonMode(GL_BACK,  static_cast<GLenum>(prev_polygon_mode[1]));
    glViewport(
        prev_viewport[0],
        prev_viewport[1],
        static_cast<GLsizei>(prev_viewport[2]),
        static_cast<GLsizei>(prev_viewport[3])
        );
    glScissor(
        prev_scissor_box[0],
        prev_scissor_box[1],
        static_cast<GLsizei>(prev_scissor_box[2]),
        static_cast<GLsizei>(prev_scissor_box[3])
        );
}


//-------------------------------------------------------------------------------------------------
// Init GLUT
//

void viewer_glut::impl::init(
        int argc,
        char** argv,
        char const* window_title,
        bool full_screen,
        int width,
        int height
        )
{
    glutInit(&argc, argv);

    glutInitDisplayMode(/*GLUT_3_2_CORE_PROFILE |*/ GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

    glutInitWindowSize(width, height);
    win_id = glutCreateWindow(window_title);

    if (full_screen)
    {
        glutFullScreen();
    }

    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    imgui_create_font_texture();

    ImGuiIO& io = ImGui::GetIO();

    io.KeyMap[ImGuiKey_Tab]         = '\t';
    io.KeyMap[ImGuiKey_LeftArrow]   = 256 + GLUT_KEY_LEFT;
    io.KeyMap[ImGuiKey_RightArrow]  = 256 + GLUT_KEY_RIGHT;
    io.KeyMap[ImGuiKey_UpArrow]     = 256 + GLUT_KEY_UP;
    io.KeyMap[ImGuiKey_DownArrow]   = 256 + GLUT_KEY_DOWN;
    io.KeyMap[ImGuiKey_PageUp]      = 256 + GLUT_KEY_PAGE_UP;
    io.KeyMap[ImGuiKey_PageDown]    = 256 + GLUT_KEY_PAGE_DOWN;
    io.KeyMap[ImGuiKey_Home]        = 256 + GLUT_KEY_HOME;
    io.KeyMap[ImGuiKey_End]         = 256 + GLUT_KEY_END;
    io.KeyMap[ImGuiKey_Insert]      = 256 + GLUT_KEY_INSERT;
    io.KeyMap[ImGuiKey_Delete]      = 127;
    io.KeyMap[ImGuiKey_Backspace]   = 8;
    io.KeyMap[ImGuiKey_Space]       = ' ';
    io.KeyMap[ImGuiKey_Enter]       = 13;
    io.KeyMap[ImGuiKey_Escape]      = 27;
    io.KeyMap[ImGuiKey_A]           = 'A';
    io.KeyMap[ImGuiKey_C]           = 'C';
    io.KeyMap[ImGuiKey_V]           = 'V';
    io.KeyMap[ImGuiKey_X]           = 'X';
    io.KeyMap[ImGuiKey_Y]           = 'Y';
    io.KeyMap[ImGuiKey_Z]           = 'Z';

    glutDisplayFunc(display_func);
    glutIdleFunc(idle_func);
    glutKeyboardFunc(keyboard_func);
    glutKeyboardUpFunc(keyboard_up_func);
    glutMotionFunc(motion_func);
    glutMouseFunc(mouse_func);
    glutPassiveMotionFunc(passive_motion_func);
    glutReshapeFunc(reshape_func);
    glutSpecialFunc(special_func);
    glutSpecialUpFunc(special_up_func);
#ifdef FREEGLUT
    glutCloseFunc(close_func);
#else
    atexit(close_func);
#endif

    GLenum error = glewInit();
    if (error != GLEW_OK)
    {
        std::string error_string("glewInit() failed: ");
        error_string.append(reinterpret_cast<char const*>(glewGetErrorString(error)));
        throw std::runtime_error(error_string);
    }
}

void viewer_glut::impl::cleanup()
{
    imgui_destroy_font_texture();
}


//-------------------------------------------------------------------------------------------------
// Dispatch to virtual event handlers
//

void viewer_glut::impl::close_func()
{
    viewer->on_close();
}

void viewer_glut::impl::display_func()
{
    ImGui::NewFrame();

    viewer->on_display();

    ImGui::Render();
    imgui_draw(ImGui::GetDrawData());

    glutSwapBuffers();
}

void viewer_glut::impl::idle_func()
{
    viewer->on_idle();
}

void viewer_glut::impl::keyboard_func(unsigned char key, int, int)
{
    // imgui
    ImGuiIO& io = ImGui::GetIO();

    if (key >= 32)
    {
        io.AddInputCharacter(key);
    }

    if (key >= 1 && key <= 26)
    {
        io.KeysDown[key] = io.KeysDown[key - 1 + 'a'] = io.KeysDown[key - 1 + 'A'] = true;
    }
    else if (key >= 'a' && key <= 'z')
    {
        io.KeysDown[key] = io.KeysDown[key - 'a' + 'A'] = true;
    }
    else if (key >= 'A' && key <= 'Z')
    {
        io.KeysDown[key] = io.KeysDown[key - 'A' + 'a'] = true;
    }
    else
    {
        io.KeysDown[key] = true;
    }

    int modifiers = glutGetModifiers();
    io.KeyCtrl = (modifiers & GLUT_ACTIVE_CTRL) != 0;
    io.KeyShift = (modifiers & GLUT_ACTIVE_SHIFT) != 0;
    io.KeyAlt = (modifiers & GLUT_ACTIVE_ALT) != 0;

    if (io.WantCaptureKeyboard)
    {
        return;
    }


    // viewer
    auto k = keyboard::map_glut_key(key);
    auto m = keyboard::map_glut_modifiers(glutGetModifiers());

    viewer->on_key_press( key_event(keyboard::KeyPress, k, m) );
}

void viewer_glut::impl::keyboard_up_func(unsigned char key, int, int)
{
    // imgui
    ImGuiIO& io = ImGui::GetIO();

    if (key >= 1 && key <= 26)
    {
        io.KeysDown[key] = io.KeysDown[key - 1 + 'a'] = io.KeysDown[key - 1 + 'A'] = false;
    }
    else if (key >= 'a' && key <= 'z')
    {
        io.KeysDown[key] = io.KeysDown[key - 'a' + 'A'] = false;
    }
    else if (key >= 'A' && key <= 'Z')
    {
        io.KeysDown[key] = io.KeysDown[key - 'A' + 'a'] = false;
    }
    else
    {
        io.KeysDown[key] = false;
    }

    int modifiers = glutGetModifiers();
    io.KeyCtrl = (modifiers & GLUT_ACTIVE_CTRL) != 0;
    io.KeyShift = (modifiers & GLUT_ACTIVE_SHIFT) != 0;
    io.KeyAlt = (modifiers & GLUT_ACTIVE_ALT) != 0;

    if (io.WantCaptureKeyboard)
    {
        return;
    }


    // viewer
    auto k = keyboard::map_glut_key(key);
    auto m = keyboard::map_glut_modifiers(glutGetModifiers());

    viewer->on_key_release( key_event(keyboard::KeyRelease, k, m) );
}

void viewer_glut::impl::motion_func(int x, int y)
{
    // imgui
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2(static_cast<float>(x), static_cast<float>(y));

    if (io.WantCaptureMouse)
    {
        return;
    }


    // viewer
    mouse::pos p = { x, y };

    mouse_event event(
            mouse::Move,
            p,
            down_button,
            keyboard::NoKey
            );

    viewer->on_mouse_move(event);
}

void viewer_glut::impl::mouse_func(int button, int state, int x, int y)
{
    // imgui
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2(static_cast<float>(x), static_cast<float>(y));
    int imgui_button = button == GLUT_LEFT_BUTTON ? 0 : button == GLUT_RIGHT_BUTTON ? 1 : 2;

    if (state == GLUT_DOWN)
    {
        io.MouseDown[imgui_button] = true;
    }
    else if (state == GLUT_UP)
    {
        io.MouseDown[imgui_button] = false;
    }


    // viewer
    mouse::pos p = { x, y };

    auto b = mouse::map_glut_button(button);
    auto m = keyboard::map_glut_modifiers(glutGetModifiers());

    if (state == GLUT_DOWN)
    {
        viewer->on_mouse_down( mouse_event(mouse::ButtonDown, p, b, m) );
        down_button = b;
    }
    else if (state == GLUT_UP)
    {
        viewer->on_mouse_up( mouse_event(mouse::ButtonUp, p, b, m) );
        down_button = mouse::NoButton;
    }
}

void viewer_glut::impl::passive_motion_func(int x, int y)
{
    // imgui
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2(static_cast<float>(x), static_cast<float>(y));

    if (io.WantCaptureMouse)
    {
        return;
    }


    // viewer
    mouse::pos p = { x, y };

    mouse_event event(
            mouse::Move,
            p,
            mouse::NoButton,
            keyboard::NoKey
            );

    viewer->on_mouse_move(event);
}

void viewer_glut::impl::reshape_func(int w, int h)
{
    // imgui
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(w), static_cast<float>(h));


    // viewer
    viewer->on_resize(w, h);
}

void viewer_glut::impl::special_func(int key, int, int)
{
    auto k = keyboard::map_glut_special(key);
    auto m = keyboard::map_glut_modifiers(glutGetModifiers());

    viewer->on_key_press( key_event(keyboard::KeyPress, k, m) );
}

void viewer_glut::impl::special_up_func(int key, int, int)
{
    auto k = keyboard::map_glut_special(key);
    auto m = keyboard::map_glut_modifiers(glutGetModifiers());

    viewer->on_key_release( key_event(keyboard::KeyRelease, k, m) );
}


//-------------------------------------------------------------------------------------------------
// Public interface
//

viewer_glut::viewer_glut(
        int width,
        int height,
        char const* window_title
        )
    : viewer_base(width, height, window_title)
    , impl_(new impl(this))
{
}

viewer_glut::~viewer_glut()
{
}

void viewer_glut::init(int argc, char** argv)
{
    viewer_base::init(argc, argv);
    impl_->init(argc, argv, window_title(), full_screen(), width(), height());
}

void viewer_glut::event_loop()
{
    glutMainLoop();
}

void viewer_glut::resize(int width, int height)
{
    viewer_base::resize(width, height);
    glutReshapeWindow(width, height);
}

void viewer_glut::swap_buffers()
{
    glutSwapBuffers();
}

void viewer_glut::toggle_full_screen()
{
    // OK to use statics, this is GLUT, anyway...

    static int win_x  = glutGet(GLUT_WINDOW_X);
    static int win_y  = glutGet(GLUT_WINDOW_Y);
    static int width  = glutGet(GLUT_WINDOW_WIDTH);
    static int height = glutGet(GLUT_WINDOW_HEIGHT);

    if (full_screen())
    {
        glutReshapeWindow( width, height );
        glutPositionWindow( win_x, win_y );
    }
    else
    {
        win_x  = glutGet(GLUT_WINDOW_X);
        win_y  = glutGet(GLUT_WINDOW_Y);
        width  = glutGet(GLUT_WINDOW_WIDTH);
        height = glutGet(GLUT_WINDOW_HEIGHT);
        glutFullScreen();
    }

    viewer_base::toggle_full_screen();
}

void viewer_glut::quit()
{
    glutDestroyWindow(impl_->win_id);
#ifdef FREEGLUT
    glutLeaveMainLoop();
    viewer_base::quit();
#else
    impl_->cleanup();
    viewer_base::quit();
    exit(EXIT_SUCCESS); // TODO
#endif
}


//-------------------------------------------------------------------------------------------------
// Event handlers
//

void viewer_glut::on_idle()
{
    glutPostRedisplay();
}
