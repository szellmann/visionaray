// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_VIEWER_QT_H
#define VSNRAY_COMMON_VIEWER_QT_H 1

#include <memory>
#include <string>

#include <QGLWidget>

#include "viewer_base.h"

namespace visionaray
{

class camera_manipulator;
class qgl_widget;
class key_event;
class mouse_event;


//-------------------------------------------------------------------------------------------------
// Windowed Qt viewer
//

class viewer_qt : public QObject, public viewer_base
{
    Q_OBJECT

public:

    viewer_qt(
            int width                   = 512,
            int height                  = 512,
            std::string window_title    = "Visionaray Qt Viewer"
            );
    virtual ~viewer_qt();

    void init(int& argc, char**& argv);

    void event_loop();
    void resize(int width, int height);
    void toggle_full_screen();

public slots:

    void call_on_display();
    void call_on_idle();
    void call_on_resize(int width, int height);
    void call_on_key_press(key_event const& event);
    void call_on_key_release(key_event const& event);
    void call_on_mouse_move(mouse_event const& event);
    void call_on_mouse_down(mouse_event const& event);
    void call_on_mouse_up(mouse_event const& event);

private:

    struct impl;
    friend struct impl;
    std::unique_ptr<impl> impl_;

};


//-------------------------------------------------------------------------------------------------
// Use qgl_widget to embed a 'viewer' as a widget into a Qt application
//

class qgl_widget : public QGLWidget, public viewer_base
{
    Q_OBJECT

signals:

    void display();
    void resize(int w, int h);
    void idle();
    void key_press(key_event const& event);
    void key_release(key_event const& event);
    void mouse_move(mouse_event const& event);
    void mouse_down(mouse_event const& event);
    void mouse_up(mouse_event const& event);

public:

    qgl_widget(QGLFormat const& format, QWidget* parent = 0);
    virtual ~qgl_widget();

private:

    struct impl;
    std::unique_ptr<impl> impl_;

    // Qt interface, use viewer_base::on_XXX() to implement event handlers
    void initializeGL();
    void paintGL();
    void resizeGL(int w, int h);
    void timerEvent(QTimerEvent* event);
    void keyPressEvent(QKeyEvent* event);
    void keyReleaseEvent(QKeyEvent* event);
    void mouseMoveEvent(QMouseEvent* event);
    void mousePressEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent* event);

};

} // visionaray

#endif // VSNRAY_COMMON_VIEWER_QT_H
