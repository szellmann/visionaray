// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_COMPRESS_H
#define VSNRAY_DETAIL_BVH_COMPRESS_H 1

namespace visionaray
{

class bvh_compressor
{
public:
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

        init(input, output);
    }

    // TODO:
    // void compress_yilitie_wide()

private:

    template <typename P1, typename N1, int W1, typename P2, typename N2, int W2>
    void init(bvh_t<P1, N1, W1> const& input, bvh_t<P2, N2, W2>& output)
    {
        output.primitives() = input.primitives();
    }

    template <typename P1, typename N1, typename I1, int W1, typename P2, typename N2, typename I2, int W2>
    void init(index_bvh_t<P1, N1, I1, W1> const& input, index_bvh_t<P2, N2, I2, W2>& output)
    {
        output.primitives() = input.primitives();
        output.indices() = input.indices();
    }
};

} // namespace visionaray

#endif // VSNRAY_DETAIL_BVH_COMPRESS_H
