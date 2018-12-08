// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cassert>
#include <cstring>

#include "make_unique.h"
#include "model.h"
#include "sg.h"

namespace visionaray
{
namespace sg
{

//-------------------------------------------------------------------------------------------------
// node
//

std::string& node::name()
{
    if (meta_data_ == nullptr)
    {
        meta_data_ = make_unique<meta_data>();
    }

    return meta_data_->name;
}

std::string const& node::name() const
{
    if (meta_data_ == nullptr)
    {
        const_cast<node*>(this)->meta_data_ = make_unique<meta_data>();
    }

    return meta_data_->name;
}

uint64_t& node::flags()
{
    if (meta_data_ == nullptr)
    {
        meta_data_ = make_unique<meta_data>();
    }

    return meta_data_->flags;
}

uint64_t const& node::flags() const
{
    if (meta_data_ == nullptr)
    {
        const_cast<node*>(this)->meta_data_ = make_unique<meta_data>();
    }

    return meta_data_->flags;
}

std::vector<std::weak_ptr<node>>& node::parents()
{
    return parents_;
}

std::vector<std::weak_ptr<node>> const& node::parents() const
{
    return parents_;
}

std::vector<std::shared_ptr<node>>& node::children()
{
    return children_;
}

std::vector<std::shared_ptr<node>> const& node::children() const
{
    return children_;
}

void node::add_child(std::shared_ptr<node> child)
{
    child->parents_.push_back(shared_from_this());
    children_.push_back(child);
}


//-------------------------------------------------------------------------------------------------
// environment_light
//

std::shared_ptr<sg::texture>& environment_light::texture()
{
    return texture_;
}

std::shared_ptr<sg::texture> const& environment_light::texture() const
{
    return texture_;
}


//-------------------------------------------------------------------------------------------------
// transform
//

transform::transform()
    : matrix_(mat4::identity())
{
}

transform::transform(mat4 matrix)
    : matrix_(matrix)
{
}

mat4& transform::matrix()
{
    return matrix_;
}

mat4 const& transform::matrix() const
{
    return matrix_;
}


#if VSNRAY_COMMON_HAVE_PTEX

//-------------------------------------------------------------------------------------------------
// ptex_texture
//

ptex_texture::ptex_texture(std::string filename, std::shared_ptr<PtexPtr<PtexCache>> cache)
    : filename_(filename)
    , cache_(cache)
{
}

std::string& ptex_texture::filename()
{
    return filename_;
}

std::string const& ptex_texture::filename() const
{
    return filename_;
}

std::shared_ptr<PtexPtr<PtexCache>>& ptex_texture::cache()
{
    return cache_;
}

std::shared_ptr<PtexPtr<PtexCache>> const& ptex_texture::cache() const
{
    return cache_;
}


#endif // VSNRAY_COMMON_HAVE_PTEX


//-------------------------------------------------------------------------------------------------
// surface_properties
//

std::shared_ptr<material>& surface_properties::material()
{
    return material_;
}

std::shared_ptr<material> const& surface_properties::material() const
{
    return material_;
}

std::vector<std::shared_ptr<sg::texture>>& surface_properties::textures()
{
    return textures_;
}

std::vector<std::shared_ptr<sg::texture>> const& surface_properties::textures() const
{
    return textures_;
}

void surface_properties::add_texture(std::shared_ptr<sg::texture> texture)
{
    textures_.push_back(texture);
}


//-------------------------------------------------------------------------------------------------
// node_visitor
//

node_visitor::node_visitor(traversal_type tt)
    : traversal_type_(tt)
{
}

void node_visitor::apply(node& n)
{
    if (traversal_type_ == TraverseChildren)
    {
        for (auto& c : n.children_)
        {
            c->accept(*this);
        }
    }
    else if (traversal_type_ == TraverseParents)
    {

        for (auto& p : n.parents_)
        {
            auto pp = p.lock();
            pp->accept(*this);
        }
    }
}

void node_visitor::apply(camera& c)
{
    apply(static_cast<node&>(c));
}

void node_visitor::apply(light& l)
{
    apply(static_cast<node&>(l));
}

void node_visitor::apply(environment_light& el)
{
    apply(static_cast<node&>(el));
}

void node_visitor::apply(transform& t)
{
    apply(static_cast<node&>(t));
}

void node_visitor::apply(surface_properties& sp)
{
    apply(static_cast<node&>(sp));
}

void node_visitor::apply(triangle_mesh& tm)
{
    apply(static_cast<node&>(tm));
}

void node_visitor::apply(sphere& s)
{
    apply(static_cast<node&>(s));
}


//-------------------------------------------------------------------------------------------------
// material
//

std::string& material::name()
{
    return name_;
}

std::string const& material::name() const
{
    return name_;
}


//-------------------------------------------------------------------------------------------------
// texture
//

texture::~texture()
{
}

std::string& texture::name()
{
    return name_;
}

std::string const& texture::name() const
{
    return name_;
}

} // sg
} // visionaray
