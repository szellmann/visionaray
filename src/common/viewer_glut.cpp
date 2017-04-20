// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdlib>
#include <stdexcept>

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

#include "input/glut.h"
#include "viewer_glut.h"


using namespace visionaray;

struct viewer_glut::impl
{
    static viewer_glut*     viewer;
    static mouse::button    down_button;
    static int              win_id;

    impl(viewer_glut* instance);

    void init(
            int argc,
            char** argv,
            std::string window_title,
            bool full_screen,
            int width,
            int height
            );

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
};

viewer_glut*    viewer_glut::impl::viewer       = nullptr;
mouse::button   viewer_glut::impl::down_button  = mouse::NoButton;
int             viewer_glut::impl::win_id       = 0;


//-------------------------------------------------------------------------------------------------
// Private implementation methods
//-------------------------------------------------------------------------------------------------

viewer_glut::impl::impl(viewer_glut* instance)
{
    viewer_glut::impl::viewer = instance;
}


//-------------------------------------------------------------------------------------------------
// Init GLUT
//

void viewer_glut::impl::init(
        int argc,
        char** argv,
        std::string window_title,
        bool full_screen,
        int width,
        int height
        )
{
    glutInit(&argc, argv);

    glutInitDisplayMode(/*GLUT_3_2_CORE_PROFILE |*/ GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

    glutInitWindowSize(width, height);
    win_id = glutCreateWindow(window_title.c_str());

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
#ifdef FREEGLUT
    glutCloseFunc(close_func);
#else
    atexit(close_func);
#endif

    if (glewInit() != GLEW_OK)
    {
        throw std::runtime_error("glewInit() failed");
    }
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
    viewer->on_display();

    glutSwapBuffers();
}

void viewer_glut::impl::idle_func()
{
    viewer->on_idle();
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
