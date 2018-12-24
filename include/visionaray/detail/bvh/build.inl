// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>

#include "../compiler.h"
#include "build_top_down.h"
#include "lbvh.h"
#include "sah.h"


namespace visionaray
{

template <typename Tree, typename P>
Tree VSNRAY_DEPRECATED build(lbvh_builder /* */, P* primitives, size_t num_prims)
{
    lbvh_builder builder;

    return builder.build<Tree>(primitives, num_prims);
}

template <typename Tree, typename P>
Tree VSNRAY_DEPRECATED build(binned_sah_builder /* */, P* primitives, size_t num_prims, bool enable_spatial_splits)
{
    binned_sah_builder builder;

    builder.enable_spatial_splits(enable_spatial_splits);
    builder.set_alpha(1.0e-5f);

    return builder.build<Tree>(primitives, num_prims);
}


//--------------------------------------------------------------------------------------------------
// Default: binned_sah builder
//

template <typename Tree, typename P>
Tree VSNRAY_DEPRECATED build(P* primitives, size_t num_prims, bool enable_spatial_splits)
{
    return build<Tree>(
            binned_sah_builder{},
            primitives,
            num_prims,
            enable_spatial_splits
            );
}


} // visionaray
