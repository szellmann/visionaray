// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/pinhole_camera.h>

#include "camera_manipulator.h"
#include "../input/mouse.h"
#include "../input/key_event.h"
#include "../input/keyboard.h"


using namespace visionaray;


fp_manipulator::fp_manipulator(pinhole_camera& cam)
    : camera_manipulator(cam)
{
}


fp_manipulator::~fp_manipulator()
{
}


void fp_manipulator::handle_key_press(key_event const& event)
{

    switch (event.key())
    {

    case keyboard::ArrowLeft:
        break;
    case keyboard::ArrowRight:
        break;
    default:
        break;

    }

    camera_manipulator::handle_key_press(event);

}
