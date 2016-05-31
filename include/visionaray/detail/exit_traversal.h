// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_EXIT_TRAVERSAL_H
#define VSNRAY_DETAIL_EXIT_TRAVERSAL_H 1

#include "macros.h"
#include "tags.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// exit_traversal
//
// A helper type to determine if ray object traversal can be terminated early.
// Applies in the special case of any-hit traversal and all rays in a ray packet
// having hit an object.
//
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// General case
//

template <traversal_type Traversal>
struct exit_traversal
{
    template <typename HR>
    VSNRAY_FUNC
    bool check(HR const& hr) const
    {
        VSNRAY_UNUSED(hr);
        return false;
    }
};


//-------------------------------------------------------------------------------------------------
// Special case any-hit
//

template <>
struct exit_traversal<AnyHit>
{
    template <typename HR>
    VSNRAY_FUNC
    bool check(HR const& hr) const
    {
        return all(hr.hit);
    }
};

} // detail
} // visionaray

#endif // VSNRAY_DETAIL_EXIT_TRAVERSAL_H
