// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <GL/glew.h>

#include <QApplication>
#include <QCloseEvent>
#include <QKeyEvent>
#include <QMainWindow>
#include <QMouseEvent>
#include <QWindow>

#include "input/qt.h"
#include "viewer_qt.h"


using namespace support;
using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// viewer_qt private implementation
//-------------------------------------------------------------------------------------------------

struct viewer_qt::impl
{
    void init(int& argc, char**& argv)
    {
        gl_format.setDoubleBuffer(true);
        gl_format.setDepth(true);
        gl_format.setRgba(true);
        gl_format.setAlpha(true);

        reset();

        app         = new QApplication(argc, argv);
        main_window = new QMainWindow(nullptr);
        gl_widget   = new qgl_widget(gl_format, main_window);

        main_window->setCentralWidget(gl_widget);
    }

   ~impl()
    {
        reset();
    }

    void reset()
    {
        delete main_window;
        delete app;
    }

    QApplication*   app         = nullptr;
    QMainWindow*    main_window = nullptr;
    qgl_widget*     gl_widget   = nullptr;
    QGLFormat       gl_format;
};


//-------------------------------------------------------------------------------------------------
// viewer_qt public interface
//-------------------------------------------------------------------------------------------------

viewer_qt::viewer_qt(
        int width,
        int height,
        char const* window_title
        )
    : viewer_base(width, height, window_title)
    , impl_(new impl)
{
}

viewer_qt::~viewer_qt()
{
}

void viewer_qt::init(int& argc, char**& argv)
{
    viewer_base::init(argc, argv);
    impl_->init(argc, argv);
}


void viewer_qt::event_loop()
{
    connect(
            impl_->gl_widget,
            SIGNAL(close()),
            this,
            SLOT(call_on_close())
            );

    connect(
            impl_->gl_widget,
            SIGNAL(display()),
            this,
            SLOT(call_on_display())
            );

    connect(
            impl_->gl_widget,
            SIGNAL(resize(int, int)),
            this,
            SLOT(call_on_resize(int, int))
            );

    connect(
            impl_->gl_widget,
            SIGNAL(idle()),
            this,
            SLOT(call_on_idle())
            );

    connect(
            impl_->gl_widget,
            SIGNAL(key_press(key_event const&)),
            this,
            SLOT(call_on_key_press(key_event const&))
            );

    connect(
            impl_->gl_widget,
            SIGNAL(key_release(key_event const&)),
            this,
            SLOT(call_on_key_release(key_event const&))
            );

    connect(
            impl_->gl_widget,
            SIGNAL(mouse_move(mouse_event const&)),
            this,
            SLOT(call_on_mouse_move(mouse_event const&))
            );

    connect(
            impl_->gl_widget,
            SIGNAL(mouse_down(mouse_event const&)),
            this,
            SLOT(call_on_mouse_down(mouse_event const&))
            );

    connect(
            impl_->gl_widget,
            SIGNAL(mouse_up(mouse_event const&)),
            this,
            SLOT(call_on_mouse_up(mouse_event const&))
            );

    impl_->main_window->resize(width(), height());
    impl_->main_window->setWindowTitle(QString(window_title()));
    impl_->main_window->show();

    impl_->app->exec();
}

void viewer_qt::resize(int width, int height)
{
    impl_->main_window->resize(width, height);
}

void viewer_qt::toggle_full_screen()
{
    if (full_screen())
    {
        impl_->main_window->showNormal();
    }
    else
    {
        impl_->main_window->showFullScreen();
    }

    viewer_base::toggle_full_screen();
}

void viewer_qt::quit()
{
    impl_->main_window->close();

    viewer_base::quit();
}


//-------------------------------------------------------------------------------------------------
// Qt slots
//

void viewer_qt::call_on_close()
{
    on_close();
}

void viewer_qt::call_on_display()
{
    on_display();
}

void viewer_qt::call_on_idle()
{
    on_idle();
}

void viewer_qt::call_on_resize(int width, int height)
{
    on_resize(width, height);
}

void viewer_qt::call_on_key_press(key_event const& event)
{
    on_key_press(event);
}

void viewer_qt::call_on_key_release(key_event const& event)
{
    on_key_release(event);
}

void viewer_qt::call_on_mouse_move(visionaray::mouse_event const& event)
{
    on_mouse_move(event);
}

void viewer_qt::call_on_mouse_down(visionaray::mouse_event const& event)
{
    on_mouse_down(event);
}

void viewer_qt::call_on_mouse_up(visionaray::mouse_event const& event)
{
    on_mouse_up(event);
}


//-------------------------------------------------------------------------------------------------
// qgl_widget
//-------------------------------------------------------------------------------------------------

struct qgl_widget::impl
{
    mouse::button down_button = mouse::NoButton;
};

qgl_widget::qgl_widget(QGLFormat const& format, QWidget* parent)
    : QGLWidget(format, parent)
    , impl_(new impl)
{
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
    startTimer(0);
}

qgl_widget::~qgl_widget()
{
}


//-------------------------------------------------------------------------------------------------
// Qt event handlers
//

void qgl_widget::initializeGL()
{
    glewInit();
}

void qgl_widget::paintGL()
{
    viewer_base::on_display();
    emit display();
}

void qgl_widget::resizeGL(int w, int h)
{
    viewer_base::on_resize(w, h);
    emit resize(w, h);
}

void qgl_widget::closeEvent(QCloseEvent* /*event*/)
{
    viewer_base::on_close();
    emit close();
}

void qgl_widget::timerEvent(QTimerEvent* /*event*/)
{
    viewer_base::on_idle();
    emit idle();
    updateGL();
}

void qgl_widget::keyPressEvent(QKeyEvent* event)
{
    auto k = keyboard::map_qt_key(event->key(), event->modifiers());
    auto m = keyboard::map_qt_modifiers(event->modifiers());
    auto e = key_event(keyboard::KeyPress, k, m);

    viewer_base::on_key_press(e);
    emit key_press(e);
}

void qgl_widget::keyReleaseEvent(QKeyEvent* event)
{
    auto k = keyboard::map_qt_key(event->key(), event->modifiers());
    auto m = keyboard::map_qt_modifiers(event->modifiers());
    auto e = key_event(keyboard::KeyRelease, k, m);

    viewer_base::on_key_press(e);
    emit key_release(e);
}

void qgl_widget::mouseMoveEvent(QMouseEvent* event)
{
    mouse::pos p = {
            event->pos().x() * devicePixelRatio(),
            event->pos().y() * devicePixelRatio()
            };

    mouse_event e(
            mouse::Move,
            p,
            impl_->down_button,
            keyboard::NoKey
            );

    viewer_base::on_mouse_move(e);
    emit mouse_move(e);
}

void qgl_widget::mousePressEvent(QMouseEvent* event)
{
    mouse::pos p = {
            event->pos().x() * devicePixelRatio(),
            event->pos().y() * devicePixelRatio()
            };

    auto b = mouse::map_qt_button(event->button());
    auto m = keyboard::map_qt_modifiers(event->modifiers());
    auto e = mouse_event(mouse::ButtonDown, p, b, m);

    viewer_base::on_mouse_down(e);
    emit mouse_down(e);
    impl_->down_button = b;
}

void qgl_widget::mouseReleaseEvent(QMouseEvent* event)
{
    mouse::pos p = {
            event->pos().x() * devicePixelRatio(),
            event->pos().y() * devicePixelRatio()
            };

    auto b = mouse::map_qt_button(event->button());
    auto m = keyboard::map_qt_modifiers(event->modifiers());
    auto e = mouse_event(mouse::ButtonUp, p, b, m);

    viewer_base::on_mouse_up(e);
    emit mouse_up(e);
    impl_->down_button = mouse::NoButton;
}
