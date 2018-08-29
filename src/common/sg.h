// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_SG_H
#define VSNRAY_COMMON_SG_H 1

#include <memory>
#include <vector>

#include <visionaray/math/forward.h>
#include <visionaray/math/matrix.h>
#include <visionaray/aligned_vector.h>

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
    virtual void apply(triangle_mesh& tm);
    virtual void apply(sphere& s);

protected:

    traversal_type traversal_type_ = TraverseChildren;
};

struct vertex
{
    vec3 pos;
    vec3 normal;
    vec3 tex_coord;
    vec4 color;
};

class node : public std::enable_shared_from_this<node>
{
public:

    friend class node_visitor;

    VSNRAY_SG_NODE

    using node_pointer = std::shared_ptr<node>;

    std::vector<node_pointer>& parents();
    std::vector<node_pointer> const& parents() const;

    std::vector<node_pointer>& children();
    std::vector<node_pointer> const& children() const;

    void add_child(node_pointer child);

protected:

    std::vector<node_pointer> parents_;
    std::vector<node_pointer> children_;

};

class transform : public node
{
public:

    VSNRAY_SG_NODE

    transform();

    transform(mat4 matrix);

    void set_matrix(mat4 matrix);
    mat4 get_matrix() const;

private:

    mat4 matrix_;

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
