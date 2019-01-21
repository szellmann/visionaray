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
#include <visionaray/point_light.h>
#include <visionaray/spectrum.h>
#include <visionaray/spot_light.h>
#include <visionaray/thin_lens_camera.h>

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
class camera;
class light;
class point_light;
class spot_light;
class environment_light;
class transform;
class surface_properties;
class triangle_mesh;
class indexed_triangle_mesh;
class sphere;

// Node visitor base class

enum traversal_type { TraverseChildren, TraverseParents };

class node_visitor
{
public:

    node_visitor() = default;
    node_visitor(traversal_type tt);

    virtual void apply(node& n);
    virtual void apply(camera& c);
    virtual void apply(light& l);
    virtual void apply(point_light& pl);
    virtual void apply(spot_light& sl);
    virtual void apply(environment_light& el);
    virtual void apply(transform& t);
    virtual void apply(surface_properties& sp);
    virtual void apply(triangle_mesh& tm);
    virtual void apply(indexed_triangle_mesh& itm);
    virtual void apply(sphere& s);

protected:

    traversal_type traversal_type_ = TraverseChildren;
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
// Wavefront obj material
//

struct obj_material : material
{
    // Ambient color
    vec3 ca = { 0.2f, 0.2f, 0.2f };

    // Diffuse color
    vec3 cd = { 0.8f, 0.8f, 0.8f };

    // Specular color
    vec3 cs = { 0.1f, 0.1f, 0.1f };

    // Emissive color
    vec3 ce = { 0.0f, 0.0f, 0.0f };

    // Reflective color
    vec3 cr = { 0.0f, 0.0f, 0.0f };

    // Index of refraction
    vec3 ior = { 1.0f, 1.0f, 1.0f };

    // Absorption
    vec3 absorption = { 0.0f, 0.0f, 0.0f };

    // Transmission
    float transmission = 0.0f;

    // Specular exponent
    float specular_exp = 32.0f;

    // Wavefront obj illumination model (default: 1 maps to plastic).
    int illum = 2;
};


//-------------------------------------------------------------------------------------------------
// Glass material
//

struct glass_material : material
{
    // Transmissive color
    vec3 ct = { 0.8f, 0.8f, 0.8f };

    // Reflective color
    vec3 cr = { 0.2f, 0.2f, 0.2f };

    // Index of refraction
    vec3 ior = { 1.0f, 1.0f, 1.0f };
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

    // Sheen
    float sheen = 0.0f;

    // Sheen tint
    float sheen_tint = 0.0f;

    // Index of refraction
    float ior = 1.0f;

    // Refractivity
    float refractive = 0.0f;

    // Roughness
    float roughness = 0.0f;


    // TODO..
};


//-------------------------------------------------------------------------------------------------
// Texture base class
//

class texture
{
public:
    virtual ~texture();

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
// Camera node
//

class camera : public node, public thin_lens_camera
{
public:

    VSNRAY_SG_NODE

};


//-------------------------------------------------------------------------------------------------
// Light base class
//

class light : public node
{
public:

    VSNRAY_SG_NODE

};


//-------------------------------------------------------------------------------------------------
// Point light node
//

class point_light : public light, public visionaray::point_light<float>
{
public:

    VSNRAY_SG_NODE

};


//-------------------------------------------------------------------------------------------------
// Spot light node
//

class spot_light : public light, public visionaray::spot_light<float>
{
public:

    VSNRAY_SG_NODE

};


//-------------------------------------------------------------------------------------------------
// Environment light node
//

class environment_light : public light
{
public:

    VSNRAY_SG_NODE

    std::shared_ptr<sg::texture>& texture();
    std::shared_ptr<sg::texture> const& texture() const;

private:

    std::shared_ptr<sg::texture> texture_;

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

    // Vertex positions 
    aligned_vector<vec3> vertices;

    // Shading normals
    aligned_vector<vec3> normals;

    // Texture coordinates
    aligned_vector<vec2> tex_coords;

    // Per-vertex colors
    aligned_vector<vector<3, unorm<8>>> colors;

    // Face IDs for Ptex
    aligned_vector<int> face_ids;

};


//-------------------------------------------------------------------------------------------------
// Indexed triangle mesh node
//

class indexed_triangle_mesh : public node
{
public:

    VSNRAY_SG_NODE

    // Indices
    aligned_vector<int> indices;

    // Vertex positions 
    aligned_vector<vec3> vertices;

    // Shading normals
    aligned_vector<vec3> normals;

    // Texture coordinates
    aligned_vector<vec2> tex_coords;

    // Per-vertex colors
    aligned_vector<vector<3, unorm<8>>> colors;

    // Face IDs for Ptex
    aligned_vector<int> face_ids;

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

} // sg
} // visionaray

#endif // VSNRAY_COMMON_SG_H
