// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BASIC_SCHED_H
#define VSNRAY_DETAIL_BASIC_SCHED_H 1

namespace visionaray
{

template <typename Backend, typename R>
class basic_sched
{
public:

    template <typename ...Args>
    explicit basic_sched(Args&&... args);

    template <typename K, typename SP>
    void frame(K kernel, SP sched_params);

    template <typename ...Args>
    void reset(Args&&... args);

private:

    Backend backend_;

};

} // visionaray

#include "basic_sched.inl"

#endif // VSNRAY_DETAIL_BASIC_SCHED_H
