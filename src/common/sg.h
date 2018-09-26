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
#include <visionaray/math/unorm.h>
#include <visionaray/math/vector.h>
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

class vertex
{
public:

    vertex() = default;

    vertex(vec3 p, vec3 n, vec2 tc, vec3 col, int fid);

    vec3 pos() const;
    vec3 normal() const;
    vec2 tex_coord() const;
    vector<4, unorm<8>> color() const;
    int face_id() const;

private:

    float data_[10];
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

    ptex_texture(std::string filename, std::shared_ptr<PtexPtr<PtexCache>> cache);

    std::string& filename();
    std::string const& filename() const;

    std::shared_ptr<PtexPtr<PtexCache>>& cache();
    std::shared_ptr<PtexPtr<PtexCache>> const& cache() const;

private:

    std::string filename_;
    std::shared_ptr<PtexPtr<PtexCache>> cache_;

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

    std::string& name();
    std::string const& name() const;

    uint64_t& flags();
    uint64_t const& flags() const;

    std::vector<std::weak_ptr<node>>& parents();
    std::vector<std::weak_ptr<node>> const& parents() const;

    std::vector<std::shared_ptr<node>>& children();
    std::vector<std::shared_ptr<node>> const& children() const;

    void add_child(std::shared_ptr<node> child);

protected:

    struct meta_data
    {
        std::string name;
        uint64_t flags = 0;
    };

    std::unique_ptr<meta_data> meta_data_ = nullptr;
    std::vector<std::weak_ptr<node>> parents_;
    std::vector<std::shared_ptr<node>> children_;

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

    using texture_iterator       = std::vector<std::shared_ptr<sg::texture>>::iterator;
    using const_texture_iterator = std::vector<std::shared_ptr<sg::texture>>::const_iterator;

    using texture_pointer        = std::vector<std::shared_ptr<sg::texture>>::pointer;
    using const_texture_pointer  = std::vector<std::shared_ptr<sg::texture>>::const_pointer;

public:

    std::shared_ptr<sg::material>& material();
    std::shared_ptr<sg::material> const& material() const;

    texture_pointer        textures();
    const_texture_pointer  textures() const;

    texture_iterator       textures_begin();
    const_texture_iterator textures_begin() const;

    texture_iterator       textures_end();
    const_texture_iterator textures_end() const;

    void add_texture(std::shared_ptr<sg::texture> texture);

    size_t num_textures() const;

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

    using index_iterator        = aligned_vector<int>::iterator;
    using const_index_iterator  = aligned_vector<int>::const_iterator;

    using index_pointer         = aligned_vector<int>::pointer;
    using const_index_pointer   = aligned_vector<int>::const_pointer;

    using vertex_iterator       = aligned_vector<vertex>::iterator;
    using const_vertex_iterator = aligned_vector<vertex>::const_iterator;

    using vertex_pointer        = aligned_vector<vertex>::pointer;
    using const_vertex_pointer  = aligned_vector<vertex>::const_pointer;

public:

    index_pointer         indices();
    const_index_pointer   indices() const;

    index_iterator        indices_begin();
    const_index_iterator  indices_begin() const;

    index_iterator        indices_end();
    const_index_iterator  indices_end() const;

    void add_index(int i);

    void resize_indices(size_t size);

    size_t num_indices() const;


    vertex_pointer        vertices();
    const_vertex_pointer  vertices() const;

    vertex_iterator       vertices_begin();
    const_vertex_iterator vertices_begin() const;

    vertex_iterator       vertices_end();
    const_vertex_iterator vertices_end() const;

    void add_vertex(vertex v);

    void resize_vertices(size_t size);

    size_t num_vertices() const;

private:

    // Triangle indices
    aligned_vector<int> indices_;

    // Triangle vertices
    aligned_vector<vertex> vertices_;

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
