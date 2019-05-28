// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_SG_IO_H
#define VSNRAY_COMMON_SG_IO_H 1

#include <memory>
#include <ostream>

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

    void apply(sg::node& n)
    {
        ++depth_;

        node_visitor::apply(n);

        --depth_;
    }

    void apply(surface_properties& sp)
    {
        ++depth_;

        for (int i = 0; i < depth_; ++i)
        {
            out_ << ' ';
        }

        if (sp.material() != nullptr)
        {
            if (auto m = std::dynamic_pointer_cast<obj_material>(sp.material()))
            {
                out_ << "obj_material, ";
            }
            else if (auto m = std::dynamic_pointer_cast<glass_material>(sp.material()))
            {
                out_ << "glass_material, ";
            }
            else if (auto m = std::dynamic_pointer_cast<disney_material>(sp.material()))
            {
                out_ << "disney_material, ";
            }

            out_ << '"' << sp.material()->name() << "\"\n";
        }

        node_visitor::apply(sp);

        --depth_;
    }

    void apply(triangle_mesh& tm)
    {
        ++depth_;

        for (int i = 0; i < depth_; ++i)
        {
            out_ << ' ';
        }

        out_ << "triangle_mesh\n";

        node_visitor::apply(tm);

        --depth_;
    }

    void apply(indexed_triangle_mesh& itm)
    {
        ++depth_;

        for (int i = 0; i < depth_; ++i)
        {
            out_ << ' ';
        }

        out_ << "indexed_triangle_mesh\n";

        node_visitor::apply(itm);

        --depth_;
    }

private:
    Stream& out_;
    int depth_ = 0;

};

} // detail

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, std::shared_ptr<node> const& n)
{
    detail::ostream_visitor<std::basic_ostream<CharT, Traits>> visitor(out);

    n->accept(visitor);

    return out;
}

} // sg

} // visionaray

#endif // VSNRAY_COMMON_SG_IO_H
