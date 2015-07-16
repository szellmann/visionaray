// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vector>

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN)

#include <AvailabilityMacros.h>

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

  #pragma GCC diagnostic ignored "-Wdeprecated"

#endif

#include <OpenGL/gl.h>
#include <GLUT/glut.h>

#else // VSNRAY_OS_DARWIN

#if defined(VSNRAY_OS_WIN32)
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glut.h>

#endif

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include "input/keyboard.h"
#include "input/mouse.h"
#include "manip/camera_manipulator.h"
#include "viewer_glut.h"


using namespace support;
using namespace visionaray;

using manipulators      = std::vector<std::shared_ptr<camera_manipulator>>;
using cmdline_options   = std::vector<std::shared_ptr<cl::OptionBase>>;

struct viewer_glut::impl
{
    static viewer_glut*     viewer;
    static manipulators     manips;
    static cmdline_options  options;
    static mouse::button    down_button;
    static mouse::pos       motion_pos;
    static mouse::pos       down_pos;
    static mouse::pos       up_pos;
    static bool             full_screen;
    static int              width;
    static int              height;
    static vec3             bgcolor;
    static std::string      window_title;
    static cl::CmdLine      cmd;

    impl(
            viewer_glut* instance,
            int width,
            int height,
            std::string window_title
            );

    void init(int argc, char** argv);

    void parse_cmd_line(int argc, char** argv);

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
};

viewer_glut*    viewer_glut::impl::viewer       = nullptr;
manipulators    viewer_glut::impl::manips       = manipulators();
cmdline_options viewer_glut::impl::options      = cmdline_options();
mouse::button   viewer_glut::impl::down_button  = mouse::NoButton;
mouse::pos      viewer_glut::impl::motion_pos   = { 0, 0 };
mouse::pos      viewer_glut::impl::down_pos     = { 0, 0 };
mouse::pos      viewer_glut::impl::up_pos       = { 0, 0 };
bool            viewer_glut::impl::full_screen  = false;
int             viewer_glut::impl::width        = 512;
int             viewer_glut::impl::height       = 512;
vec3            viewer_glut::impl::bgcolor      = { 0.1f, 0.4f, 1.0f };
std::string     viewer_glut::impl::window_title = "";
cl::CmdLine     viewer_glut::impl::cmd          = cl::CmdLine();


//-------------------------------------------------------------------------------------------------
// Private implementation methods
//

viewer_glut::impl::impl(
        viewer_glut* instance,
        int width,
        int height,
        std::string window_title
        )
{
    viewer_glut::impl::viewer       = instance;
    viewer_glut::impl::width        = width;
    viewer_glut::impl::height       = height;
    viewer_glut::impl::window_title = window_title;


    // add default options (-fullscreen, -width, -height, -bgcolor)

    options.emplace_back( cl::makeOption<bool&>(
        cl::Parser<>(),
        "fullscreen",
        cl::Desc("Full screen window"),
        cl::ArgDisallowed,
        cl::init(viewer_glut::impl::full_screen)
        ) );

    options.emplace_back( cl::makeOption<int&>(
        cl::Parser<>(),
        "width",
        cl::Desc("Window width"),
        cl::ArgRequired,
        cl::init(viewer_glut::impl::width)
        ) );

    options.emplace_back( cl::makeOption<int&>(
        cl::Parser<>(),
        "height",
        cl::Desc("Window height"),
        cl::ArgRequired,
        cl::init(viewer_glut::impl::height)
        ) );

    options.emplace_back( cl::makeOption<vec3&, cl::ScalarType>(
        [&](StringRef name, StringRef /*arg*/, vec3& value)
        {
            cl::Parser<>()(name + "-r", cmd.bump(), value.x);
            cl::Parser<>()(name + "-g", cmd.bump(), value.y);
            cl::Parser<>()(name + "-b", cmd.bump(), value.z);
        },
        "bgcolor",
        cl::Desc("Background color"),
        cl::ArgDisallowed,
        cl::init(viewer_glut::impl::bgcolor)
        ) );
}


//-------------------------------------------------------------------------------------------------
// Init GLUT
//

void viewer_glut::impl::init(int argc, char** argv)
{
    parse_cmd_line(argc, argv);

    glutInit(&argc, argv);

    glutInitDisplayMode(/*GLUT_3_2_CORE_PROFILE |*/ GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

    glutInitWindowSize(width, height);
    glutCreateWindow(window_title.c_str());

    if (full_screen)
    {
        glutFullScreen();
    }

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
}


//-------------------------------------------------------------------------------------------------
// Parse default command line options
//

void viewer_glut::impl::parse_cmd_line(int argc, char** argv)
{
    for (auto& opt : options)
    {
        cmd.add(*opt);
    }

    auto args = std::vector<std::string>(argv + 1, argv + argc);
    cl::expandWildcards(args);
    cl::expandResponseFiles(args, cl::TokenizeUnix());

    cmd.parse(args);
}


//-------------------------------------------------------------------------------------------------
// Dispatch to virtual event handlers
//

void viewer_glut::impl::display_func()
{
    viewer->on_display();
}

void viewer_glut::impl::idle_func()
{
    viewer->on_idle();
}

void viewer_glut::impl::motion_func(int x, int y)
{
    mouse::pos p = { x, y };

    mouse_event event(
            mouse::Move,
            p,
            down_button,
            keyboard::NoKey
            );

    viewer->on_mouse_move(event);
}

void viewer_glut::impl::keyboard_func(unsigned char key, int, int)
{
    auto k = keyboard::map_glut_key(key);
    auto m = keyboard::map_glut_modifiers(glutGetModifiers());

    viewer->on_key_press( key_event(keyboard::KeyPress, k, m) );
}

void viewer_glut::impl::keyboard_up_func(unsigned char key, int, int)
{
    auto k = keyboard::map_glut_key(key);
    auto m = keyboard::map_glut_modifiers(glutGetModifiers());

    viewer->on_key_release( key_event(keyboard::KeyRelease, k, m) );
}

void viewer_glut::impl::mouse_func(int button, int state, int x, int y)
{
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
    width = w;
    height = h;
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
        std::string window_title
        )
    : impl_(new impl(this, width, height, window_title))
{
}

viewer_glut::~viewer_glut()
{
}

void viewer_glut::init(int argc, char** argv)
{
    impl_->init(argc, argv);
}

void viewer_glut::add_manipulator( std::shared_ptr<camera_manipulator> manip )
{
    impl_->manips.push_back(manip);
}

void viewer_glut::add_cmdline_option( std::shared_ptr<cl::OptionBase> option )
{
    impl_->options.emplace_back(option);
}

void viewer_glut::event_loop()
{
    glutMainLoop();
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

    if (impl_->full_screen)
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

    impl_->full_screen = !impl_->full_screen;
}

int viewer_glut::width()
{
    return impl_->width;
}

int viewer_glut::height()
{
    return impl_->height;
}

vec3 viewer_glut::background_color() const
{
    return impl_->bgcolor;
}


//-------------------------------------------------------------------------------------------------
// Event handlers
//

void viewer_glut::on_display()
{
}

void viewer_glut::on_idle()
{
    glutPostRedisplay();
}

void viewer_glut::on_key_press(visionaray::key_event const& event)
{
    if (event.key() == keyboard::F5)
    {
        toggle_full_screen();
    }

    if (event.key() == keyboard::Escape && impl_->full_screen)
    {
        toggle_full_screen();
    }

    for (auto& manip : impl_->manips)
    {
        manip->handle_key_press(event);
    }
}

void viewer_glut::on_key_release(visionaray::key_event const& event)
{
    for (auto& manip : impl_->manips)
    {
        manip->handle_key_release(event);
    }
}

void viewer_glut::on_mouse_move(visionaray::mouse_event const& event)
{
    for (auto& manip : impl_->manips)
    {
        manip->handle_mouse_move(event);
    }
}

void viewer_glut::on_mouse_down(visionaray::mouse_event const& event)
{
    for (auto& manip : impl_->manips)
    {
        manip->handle_mouse_down(event);
    }
}

void viewer_glut::on_mouse_up(visionaray::mouse_event const& event)
{
    for (auto& manip : impl_->manips)
    {
        manip->handle_mouse_up(event);
    }
}

void viewer_glut::on_resize(int w, int h)
{
    glViewport(0, 0, w, h);
}
