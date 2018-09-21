// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_SG_H
#define VSNRAY_COMMON_SG_H 1

#include <common/config.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <visionaray/texture/detail/texture_common.h> // detail!
#include <visionaray/math/forward.h>
#include <visionaray/math/matrix.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/spectrum.h>

#if VSNRAY_COMMON_HAVE_PTEX
#include "ptex.h"
#endif

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
    int face_id;
};


//-------------------------------------------------------------------------------------------------
// Material base class
//

class material
{
public:
    virtual ~material() {}

    std::string& name();
    std::string const& name() const;

private:

    std::string name_;
};


//-------------------------------------------------------------------------------------------------
// Disney principled material
//

struct disney_material : material
{
    // Base color
    vec4 base_color = vec4(0.0f);

    // Specular transmission
    float spec_trans = 0.0f;

    // Index of refraction
    float ior = 1.0f;

    // Refractivity
    float refractive = 0.0f;


    // TODO..
};


//-------------------------------------------------------------------------------------------------
// Texture base class
//

class texture
{
public:
    virtual ~texture() {}

    std::string& name();
    std::string const& name() const;

private:

    std::string name_;
};


//-------------------------------------------------------------------------------------------------
// 2D texture class
//

template <typename T>
class texture2d : public texture, public texture_base<T, 2>
{
public:

    using ref_type = texture_ref_base<T, 2>;
    using value_type = T;

public:

    void resize(int w, int h)
    {
        width_ = w;
        height_ = h;

        // Resize data vector from base
        texture_base<T, 2>::data_.resize(w * h);
    }

    int width() const { return width_; }
    int height() const { return height_; }

private:

    int width_;
    int height_;

};

#if VSNRAY_COMMON_HAVE_PTEX

//-------------------------------------------------------------------------------------------------
// WDAS Ptex node
//

class ptex_texture : public texture
{
public:

    using ref_type = PtexPtr<PtexTexture>;

public:

    ptex_texture(PtexPtr<PtexTexture>& texture);

    PtexPtr<PtexTexture> const& get() const;

private:

    PtexPtr<PtexTexture> texture_;

};

#endif // VSNRAY_COMMON_HAVE_PTEX


//-------------------------------------------------------------------------------------------------
// Node base class
//

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

    struct meta_data
    {
        std::string name;
    };

    std::unique_ptr<meta_data> meta_data_ = nullptr;
    std::vector<node_pointer> parents_;
    std::vector<node_pointer> children_;

    uint64_t flags_ = 0;

};


//-------------------------------------------------------------------------------------------------
// Transform node
//

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

    std::vector<std::shared_ptr<sg::texture>>& textures();
    std::vector<std::shared_ptr<sg::texture>> const& textures() const;

    void add_texture(std::shared_ptr<sg::texture> texture);

private:

    // Material
    std::shared_ptr<sg::material> material_ = nullptr;

    // List of textures with user definable interpretation (e.g. bump, diffuse, roughness, etc.)
    std::vector<std::shared_ptr<sg::texture>> textures_;
};


//-------------------------------------------------------------------------------------------------
// Triangle mesh node
//

class triangle_mesh : public node
{
public:

    VSNRAY_SG_NODE

    // Triangle indices
    aligned_vector<int> indices;

    // Triangle vertices
    aligned_vector<vertex> vertices;

};


//-------------------------------------------------------------------------------------------------
// Sphere node
//

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
