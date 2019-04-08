// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <cassert>

#if VSNRAY_COMMON_HAVE_3DCONNEXIONCLIENT
#include <3DConnexionClient/ConnexionClient.h>
#include <3DConnexionClient/ConnexionClientAPI.h>
#endif

#include "space_mouse.h"

#if VSNRAY_COMMON_HAVE_3DCONNEXIONCLIENT

// Import weak so we can later check if the
// driver is loaded by checking for NULL
extern int16_t SetConnexionHandlers(
        ConnexionMessageHandlerProc messageHandler,
        ConnexionAddedHandlerProc   addedHandler,
        ConnexionRemovedHandlerProc removedHandler,
        bool useSeparateThread
        ) __attribute__((weak_import));

namespace visionaray
{
namespace space_mouse
{

static uint16_t client_id = -1;

static event_callback event_callbacks[EventTypeCount] = { nullptr };

static void message_handler(unsigned product_id, unsigned message_type, void* message_arg)
{
    switch (message_type)
    {
    case kConnexionMsgDeviceState:
    {
        ConnexionDeviceState* state = reinterpret_cast<ConnexionDeviceState*>(message_arg);

        if (state->client == client_id)
        {
            switch (state->command)
            {
            case kConnexionCmdHandleAxis:
            {
                // Call translation handler
                if (event_callbacks[Translation] != nullptr && (state->axis[0] || state->axis[1] || state->axis[2]))
                {
                    event_callbacks[Translation](
                            space_mouse_event(Translation, pos(state->axis[0], state->axis[1], state->axis[2]))
                            );
                }

                // Call rotation handler
                if (event_callbacks[Rotation] != nullptr && (state->axis[3] || state->axis[4] || state->axis[5]))
                {
                    event_callbacks[Rotation](
                            space_mouse_event(Rotation, pos(state->axis[3], state->axis[4], state->axis[5]))
                            );
                }
                break;
            }
            case kConnexionCmdHandleButtons:
            {
                int buttons = 0;

                for (int bits = 0x00000001; bits != 0x10000000; bits <<= 1)
                {
                    if (state->buttons & bits)
                    {
                        buttons |= bits;
                    }
                }

                if (event_callbacks[Button] != nullptr)
                {
                    event_callbacks[Button](space_mouse_event(Button, buttons));
                }

                // TODO
                break;
            }
            }
        }
    }
    default:
        break;
    }
}

bool init()
{
    if (SetConnexionHandlers == 0)
    {
        // Driver not loaded
        return false;
    }

    // Register callback
    int16_t err = SetConnexionHandlers(message_handler, 0, 0, false);

    // Take space mouse over system-wide
    client_id = RegisterConnexionClient(
            'vray',
            0,
            kConnexionClientModeTakeOver,
            kConnexionMaskAll
            );

    return true;
}

void register_event_callback(event_type type, event_callback cb)
{
    assert(type < EventTypeCount);

    event_callbacks[type] = cb;
}

void cleanup()
{
	if (SetConnexionHandlers != 0)
	{
        if (client_id)
        {
            UnregisterConnexionClient(client_id);
        }

        CleanupConnexionHandlers();
	}
}

} // space_mouse
} // visionaray

#else

//-------------------------------------------------------------------------------------------------
// No 3DconnexionClient API
//

namespace visionaray
{
namespace space_mouse
{

bool init() { return false; }
void register_event_callback(event_type, event_callback) {}
void cleanup() {}

} // space_mouse
} // visionaray

#endif
