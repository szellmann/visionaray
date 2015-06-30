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

#include <GL/gl.h>
#include <GL/glut.h>

#endif

#include "input/keyboard.h"
#include "input/mouse.h"
#include "manip/camera_manipulator.h"
#include "viewer_glut.h"


using namespace visionaray;

using manipulators = std::vector<std::shared_ptr<camera_manipulator>>;

struct viewer_glut::impl
{
    static viewer_glut*     viewer;
    static manipulators     manips;
    static mouse::button    down_button;
    static mouse::pos       motion_pos;
    static mouse::pos       down_pos;
    static mouse::pos       up_pos;
    static int              width;
    static int              height;

    impl(
            viewer_glut* instance,
            int width,
            int height,
            std::string window_title,
            int argc,
            char** argv
            );

    static void display_func();
    static void idle_func();
    static void keyboard_func(unsigned char key, int, int);
    static void motion_func(int x, int y);
    static void mouse_func(int button, int state, int x, int y);
    static void passive_motion_func(int x, int y);
    static void reshape_func(int w, int h);
};

viewer_glut*    viewer_glut::impl::viewer       = nullptr;
manipulators    viewer_glut::impl::manips       = manipulators();
mouse::button   viewer_glut::impl::down_button  = mouse::NoButton;
mouse::pos      viewer_glut::impl::motion_pos   = { 0, 0 };
mouse::pos      viewer_glut::impl::down_pos     = { 0, 0 };
mouse::pos      viewer_glut::impl::up_pos       = { 0, 0 };
int             viewer_glut::impl::width        = 512;
int             viewer_glut::impl::height       = 512;


//-------------------------------------------------------------------------------------------------
// Init GLUT
//

viewer_glut::impl::impl(
        viewer_glut* instance,
        int width,
        int height,
        std::string window_title,
        int argc,
        char** argv
        )
{
    viewer = instance;

    glutInit(&argc, argv);

    glutInitDisplayMode(/*GLUT_3_2_CORE_PROFILE |*/ GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

    glutInitWindowSize(width, height);
    glutCreateWindow(window_title.c_str());

    glutDisplayFunc(display_func);
    glutIdleFunc(idle_func);
    glutKeyboardFunc(keyboard_func);
    glutMotionFunc(motion_func);
    glutMouseFunc(mouse_func);
    glutPassiveMotionFunc(passive_motion_func);
    glutReshapeFunc(reshape_func);
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
            mouse::NoButton,
            p,
            down_button,
            keyboard::NoKey
            );

    viewer->on_mouse_move(event);
}

void viewer_glut::impl::keyboard_func(unsigned char key, int, int)
{
    viewer->on_key_press(key);
}

void viewer_glut::impl::mouse_func(int button, int state, int x, int y)
{
    mouse::button b = mouse::map_glut_button(button);
    mouse::pos p = { x, y };

    if (state == GLUT_DOWN)
    {
        viewer->on_mouse_down( mouse_event(mouse::ButtonDown, b, p) );
        down_button = b;
    }
    else if (state == GLUT_UP)
    {
        viewer->on_mouse_up( mouse_event(mouse::ButtonUp, b, p) );
        down_button = mouse::NoButton;
    }
}

void viewer_glut::impl::passive_motion_func(int x, int y)
{
    mouse::pos p = { x, y };

    mouse_event event(
            mouse::Move,
            mouse::NoButton,
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


//-------------------------------------------------------------------------------------------------
// Public interface
//

viewer_glut::viewer_glut(
        int width,
        int height,
        std::string window_title,
        int argc,
        char** argv
        )
    : impl_(new impl(this, width, height, window_title, argc, argv))
{
}

viewer_glut::viewer_glut(std::string window_title, int argc, char** argv)
    : impl_(new impl(this, 512, 512, window_title, argc, argv))
{
}

viewer_glut::viewer_glut(int argc, char** argv)
    : impl_(new impl(this, 512, 512, "Visionaray GLUT Viewer", argc, argv))
{
}

viewer_glut::~viewer_glut()
{
}

void viewer_glut::add_manipulator( std::shared_ptr<camera_manipulator> manip )
{
    impl_->manips.push_back(manip);
}

void viewer_glut::event_loop()
{
    glutMainLoop();
}

void viewer_glut::swap_buffers()
{
    glutSwapBuffers();
}

int viewer_glut::width()
{
    return impl_->width;
}

int viewer_glut::height()
{
    return impl_->height;
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

void viewer_glut::on_key_press(unsigned char key)
{
    VSNRAY_UNUSED(key);
}

void viewer_glut::on_mouse_move(mouse_event const& event)
{
    for (auto& manip : impl_->manips)
    {
        manip->handle_mouse_move(event);
    }
}

void viewer_glut::on_mouse_down(mouse_event const& event)
{
    for (auto& manip : impl_->manips)
    {
        manip->handle_mouse_down(event);
    }
}

void viewer_glut::on_mouse_up(mouse_event const& event)
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
