// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_SG_IO_H
#define VSNRAY_COMMON_SG_IO_H 1

#include <memory>
#include <ostream>

#include <visionaray/math/io.h>

#include "../sg.h"

namespace visionaray
{
namespace sg
{
namespace detail
{

template <typename Stream>
class ostream_visitor : public node_visitor
{
public:
    using node_visitor::apply;

    ostream_visitor(Stream& out)
        : out_(out)
    {
    }

    void apply(surface_properties& sp)
    {
        ++depth_;

        indent();

        out_ << "surface_properties\n";

        if (sp.material() != nullptr)
        {
            indent();

            if (auto m = std::dynamic_pointer_cast<obj_material>(sp.material()))
            {
                out_ << "obj_material, ce: " << m->ce << '\n';
            }
            else if (auto m = std::dynamic_pointer_cast<glass_material>(sp.material()))
            {
                out_ << "glass_material\n";
            }
            else if (auto m = std::dynamic_pointer_cast<disney_material>(sp.material()))
            {
                out_ << "disney_material\n";
            }

            if (sp.material()->name() != "")
            {
                out_ << ", \"" << sp.material()->name() << "\"\n";
            }
        }

        node_visitor::apply(sp);

        --depth_;
    }

    void apply(transform& tr)
    {
        ++depth_;

        indent();

        out_ << "transform, matrix: " << tr.matrix() << '\n';

        node_visitor::apply(tr);

        --depth_;
    }

    void apply(triangle_mesh& tm)
    {
        ++depth_;

        indent();

        out_ << "triangle_mesh, vertices: " << tm.vertices.size() << '\n';

        node_visitor::apply(tm);

        --depth_;
    }

    void apply(indexed_triangle_mesh& itm)
    {
        ++depth_;

        indent();

        out_ << "indexed_triangle_mesh, vertex indices: " << itm.vertex_indices.size() << '\n';

        node_visitor::apply(itm);

        --depth_;
    }

private:
    Stream& out_;
    int depth_ = 0;

    void indent()
    {
        for (int i = 0; i < depth_; ++i)
        {
            out_ << ' ';
        }
    }
};

} // detail

template <typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, std::shared_ptr<node> const& n)
{
    detail::ostream_visitor<std::basic_ostream<CharT, Traits>> visitor(out);

    n->accept(visitor);

    return out;
}

} // sg

} // visionaray

#endif // VSNRAY_COMMON_SG_IO_H
