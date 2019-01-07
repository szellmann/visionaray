// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TAGS_H
#define VSNRAY_DETAIL_TAGS_H 1

#include <type_traits>


//-------------------------------------------------------------------------------------------------
// Tags for internal use
//

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Traversal types
//

enum traversal_type { AnyHit, ClosestHit, MultiHit };

using any_hit_tag     = std::integral_constant<int, AnyHit>;
using closest_hit_tag = std::integral_constant<int, ClosestHit>;
using multi_hit_tag   = std::integral_constant<int, MultiHit>;


//-------------------------------------------------------------------------------------------------
// Misc.
//

struct have_intersector_tag {};

} // detail
} // visionaray

#endif // VSNRAY_DETAIL_TAGS_H
