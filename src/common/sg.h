// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_SG_H
#define VSNRAY_COMMON_SG_H 1

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <visionaray/math/forward.h>
#include <visionaray/math/matrix.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/spectrum.h>

#define VSNRAY_SG_NODE                                                          \
    virtual void accept(node_visitor& visitor)                                  \
    {                                                                           \
        visitor.apply(*this);                                                   \
    }

namespace visionaray
{

class model;

namespace sg
{

class node;
class transform;
class surface_properties;
class triangle_mesh;
class sphere;

// Node visitor base class

enum traversal_type { TraverseChildren, TraverseParents };

class node_visitor
{
public:

    node_visitor() = default;
    node_visitor(traversal_type tt);

    virtual void apply(node& n);
    virtual void apply(transform& t);
    virtual void apply(surface_properties& sp);
    virtual void apply(triangle_mesh& tm);
    virtual void apply(sphere& s);

protected:

    traversal_type traversal_type_ = TraverseChildren;
};

struct vertex
{
    vec3 pos;
    vec3 normal;
    vec2 tex_coord;
    vec4 color;
};

struct material
{
    virtual ~material() {}
};


//-------------------------------------------------------------------------------------------------
// Disney principled material
//

struct disney_material : material
{
    vec4 base_color;
    // TODO..
};

class node : public std::enable_shared_from_this<node>
{
public:

    friend class node_visitor;

    VSNRAY_SG_NODE

    using node_pointer = std::shared_ptr<node>;

    std::string& name();
    std::string const& name() const;

    std::vector<node_pointer>& parents();
    std::vector<node_pointer> const& parents() const;

    std::vector<node_pointer>& children();
    std::vector<node_pointer> const& children() const;

    uint64_t& flags();
    uint64_t const& flags() const;

    void add_child(node_pointer child);

protected:

    std::string name_;
    std::vector<node_pointer> parents_;
    std::vector<node_pointer> children_;

    uint64_t flags_ = 0;

};

class transform : public node
{
public:

    VSNRAY_SG_NODE

    transform();

    transform(mat4 matrix);

    mat4& matrix();
    mat4 const& matrix() const;

private:

    mat4 matrix_;

};


//-------------------------------------------------------------------------------------------------
// Surface properties node
//

class surface_properties : public node
{
public:

    VSNRAY_SG_NODE

    std::shared_ptr<sg::material>& material();
    std::shared_ptr<sg::material> const& material() const;

private:

    std::shared_ptr<sg::material> material_ = nullptr;
};


class triangle_mesh : public node
{
public:

    VSNRAY_SG_NODE

    // Triangle indices
    aligned_vector<int> indices;

    // Triangle vertices
    aligned_vector<vertex> vertices;

};


class sphere : public node
{
public:

    VSNRAY_SG_NODE

    // Unit sphere, centered at (0,0,0) with radius (1,1,1)
    // Use transform node to change position and scale
};


// Flatten scene graph and create lists that are compatible with kernels
void flatten(model& mod, node& root);


} // sg
} // visionaray

#endif // VSNRAY_COMMON_SG_H
