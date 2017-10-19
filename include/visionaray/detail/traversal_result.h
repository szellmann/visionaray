// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TRAVERSAL_RESULT_H
#define VSNRAY_DETAIL_TRAVERSAL_RESULT_H 1

#include <cstddef>

#include "../array.h"
#include "tags.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// traversal_result
//
// Trait class to determine the result type of traversal functions.
// In the general case, this is simply a hit_record.
// In the case of multi_hit, the traversal result is an array of hit records.
//

template <typename HR, traversal_type Traversal, size_t MaxHits>
struct traversal_result
{
    using type = HR;
};

template <typename HR, size_t MaxHits>
struct traversal_result<HR, MultiHit, MaxHits>
{
    using type = array<HR, MaxHits>;
};

} // detail
} // visionaray

#endif // VSNRAY_DETAIL_TRAVERSAL_RESULT_H
