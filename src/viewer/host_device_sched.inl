// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <thread>
#include <utility>

namespace visionaray
{

host_device_sched::host_device_sched(render_state const& state)
    : state_(state)
    , host_sched_(std::thread::hardware_concurrency())
#ifdef __CUDACC__
    , device_sched_(8, 8)
#endif
{
}

template <typename ...Args>
void host_device_sched::frame(Args&&... args)
{
    if (state_.mode == render_state::CPU)
    {
        host_sched_.frame(std::forward<Args>(args)...);
    }
    else
    {
        device_sched_.frame(std::forward<Args>(args)...);
    }
}

} // visionaray
