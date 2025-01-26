// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_COMPRESS_H
#define VSNRAY_DETAIL_BVH_COMPRESS_H 1

namespace visionaray
{

struct bvh_compressor
{
    template <typename Input, typename Output>
    void compress(Input const& input, Output& output)
    {
        static_assert(Input::Width == Output::Width, "Type mismatch");

        auto const& input_nodes = input.nodes();
        auto& output_nodes = output.nodes();
        output_nodes.resize(input.num_nodes());

        for (size_t i = 0; i < input.num_nodes(); ++i)
        {
            output_nodes[i].init(input_nodes[i]);
        }

        output.primitives() = input.primitives();
        output.indices() = input.indices();
    }

    // TODO:
    // void compress_yilitie_wide()
};

} // namespace visionaray

#endif // VSNRAY_DETAIL_BVH_COMPRESS_H
