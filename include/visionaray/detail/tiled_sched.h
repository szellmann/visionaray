// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TILED_SCHED_H
#define VSNRAY_DETAIL_TILED_SCHED_H

#include <memory>

#include <visionaray/render_target.h>

namespace visionaray
{

template <typename R>
class tiled_sched
{
public:

    explicit tiled_sched(unsigned num_threads);
   ~tiled_sched();

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params, unsigned frame_num = 0);

    void set_num_threads(unsigned num_threads);
    unsigned get_num_threads() const;

private:

    struct impl;
    std::unique_ptr<impl> impl_;

};

} // visionaray

#include "tiled_sched.inl"

#endif // VSNRAY_DETAIL_TILED_SCHED_H
