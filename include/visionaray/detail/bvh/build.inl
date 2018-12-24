// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>

#include "build_top_down.h"
#include "lbvh.h"
#include "sah.h"


namespace visionaray
{


template <typename Tree, typename P>
Tree build(detail::lbvh_builder /* */, P* primitives, size_t num_prims)
{
    Tree tree(primitives, num_prims);

    detail::lbvh_builder builder;

    builder.build(tree, primitives, primitives + num_prims);

    return tree;
}


template <typename Tree, typename P>
Tree build(detail::binned_sah_builder /* */, P* primitives, size_t num_prims, bool enable_spatial_splits)
{
    Tree tree(primitives, num_prims);

    detail::binned_sah_builder builder;

    builder.enable_spatial_splits(enable_spatial_splits);
    builder.set_alpha(1.0e-5f);

    builder.build(tree, primitives, primitives + num_prims);

    return tree;
}


//--------------------------------------------------------------------------------------------------
// Default: binned_sah builder
//

template <typename Tree, typename P>
Tree build(P* primitives, size_t num_prims, bool enable_spatial_splits)
{
    return build<Tree>(
            detail::binned_sah_builder{},
            primitives,
            num_prims,
            enable_spatial_splits
            );
}


} // visionaray
