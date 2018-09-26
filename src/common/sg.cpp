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

surface_properties::texture_pointer surface_properties::textures()
{
    return textures_.data();
}

surface_properties::const_texture_pointer surface_properties::textures() const
{
    return textures_.data();
}

surface_properties::texture_iterator surface_properties::textures_begin()
{
    return textures_.begin();
}

surface_properties::const_texture_iterator surface_properties::textures_begin() const
{
    return textures_.cbegin();
}

surface_properties::texture_iterator surface_properties::textures_end()
{
    return textures_.end();
}

surface_properties::const_texture_iterator surface_properties::textures_end() const
{
    return textures_.cend();
}

void surface_properties::add_texture(std::shared_ptr<sg::texture> texture)
{
    textures_.push_back(texture);
}

size_t surface_properties::num_textures() const
{
    return textures_.size();
}


//-------------------------------------------------------------------------------------------------
// triangle_mesh
//

triangle_mesh::index_pointer triangle_mesh::indices()
{
    return indices_.data();
}

triangle_mesh::const_index_pointer triangle_mesh::indices() const
{
    return indices_.data();
}

triangle_mesh::index_iterator triangle_mesh::indices_begin()
{
    return indices_.begin();
}

triangle_mesh::const_index_iterator triangle_mesh::indices_begin() const
{
    return indices_.cbegin();
}

triangle_mesh::index_iterator triangle_mesh::indices_end()
{
    return indices_.end();
}

triangle_mesh::const_index_iterator triangle_mesh::indices_end() const
{
    return indices_.cend();
}

void triangle_mesh::add_index(int i)
{
    indices_.push_back(i);
}

void triangle_mesh::resize_indices(size_t size)
{
    indices_.resize(size);
}

size_t triangle_mesh::num_indices() const
{
    return indices_.size();
}

triangle_mesh::vertex_pointer triangle_mesh::vertices()
{
    return vertices_.data();
}

triangle_mesh::const_vertex_pointer triangle_mesh::vertices() const
{
    return vertices_.data();
}

triangle_mesh::vertex_iterator triangle_mesh::vertices_begin()
{
    return vertices_.begin();
}

triangle_mesh::const_vertex_iterator triangle_mesh::vertices_begin() const
{
    return vertices_.cbegin();
}

triangle_mesh::vertex_iterator triangle_mesh::vertices_end()
{
    return vertices_.end();
}

triangle_mesh::const_vertex_iterator triangle_mesh::vertices_end() const
{
    return vertices_.cend();
}

void triangle_mesh::add_vertex(vertex v)
{
    vertices_.push_back(v);
}

void triangle_mesh::resize_vertices(size_t size)
{
    vertices_.resize(size);
}

size_t triangle_mesh::num_vertices() const
{
    return vertices_.size();
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
// vertex
//

vertex::vertex(vec3 p, vec3 n, vec2 tc, vec3 col, int fid)
{
    std::memcpy(data_, p.data(), sizeof(float) * 3);
    std::memcpy(data_ + 3, n.data(), sizeof(float) * 3);
    std::memcpy(data_ + 6, tc.data(), sizeof(float) * 2);
    std::memcpy(data_ + 8, col.data(), sizeof(vector<4, unorm<8>>));
    std::memcpy(data_ + 9, &fid, sizeof(int));
}

vec3 vertex::pos() const
{
    return vec3(data_);
}

vec3 vertex::normal() const
{
    return vec3(data_ + 3);
}

vec2 vertex::tex_coord() const
{
    return vec2(data_ + 6);
}

vector<4, unorm<8>> vertex::color() const
{
    return *reinterpret_cast<vector<4, unorm<8>> const*>(data_ + 8);
}

int vertex::face_id() const
{
    return *reinterpret_cast<int const*>(data_ + 9);
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

std::string& texture::name()
{
    return name_;
}

std::string const& texture::name() const
{
    return name_;
}


//-------------------------------------------------------------------------------------------------
// flatten
//

struct flatten_visitor : node_visitor
{
    using node_visitor::apply;

    flatten_visitor(model& mod)
        : model_(mod)
    {
    }

    void apply(transform& t)
    {
        mat4 prev = current_transform;

        current_transform = current_transform * t.matrix();

        node_visitor::apply(t);

        current_transform = prev;
    }

    void apply(surface_properties& sp)
    {
        unsigned prev = current_geom_id;

        if (sp.material())
        {
            auto it = std::find(materials.begin(), materials.end(), sp.material());
            if (it == materials.end())
            {
                current_geom_id = static_cast<unsigned>(materials.size());
                materials.push_back(sp.material());
            }
            else
            {
                current_geom_id = static_cast<unsigned>(std::distance(materials.begin(), it));
            }
        }

        node_visitor::apply(sp);

        current_geom_id = prev;
    }

    void apply(triangle_mesh& tm)
    {
        // Matrix to transform normals
        mat4 trans_inv = inverse(transpose(current_transform));

        assert(tm.num_indices() % 3 == 0);

        size_t first_primitive = model_.primitives.size();
        model_.primitives.resize(model_.primitives.size() + tm.num_indices() / 3);

        size_t first_shading_normal = model_.shading_normals.size();
        model_.shading_normals.resize(model_.shading_normals.size() + tm.num_indices());

        size_t first_geometric_normal = model_.geometric_normals.size();
        model_.geometric_normals.resize(model_.geometric_normals.size() + tm.num_indices() / 3);

        size_t first_tex_coord = model_.tex_coords.size();
        model_.tex_coords.resize(model_.tex_coords.size() + tm.num_indices());

        for (size_t i = 0; i < tm.num_indices(); i += 3)
        {
            vec3 v1 = tm.vertices()[tm.indices()[i]].pos();
            vec3 v2 = tm.vertices()[tm.indices()[i + 1]].pos();
            vec3 v3 = tm.vertices()[tm.indices()[i + 2]].pos();

            vec3 n1 = tm.vertices()[tm.indices()[i]].normal();
            vec3 n2 = tm.vertices()[tm.indices()[i + 1]].normal();
            vec3 n3 = tm.vertices()[tm.indices()[i + 2]].normal();

            vec3 gn = normalize(cross(v2 - v1, v3 - v1));

            vec2 tc1 = tm.vertices()[tm.indices()[i]].tex_coord();
            vec2 tc2 = tm.vertices()[tm.indices()[i + 1]].tex_coord();
            vec2 tc3 = tm.vertices()[tm.indices()[i + 2]].tex_coord();

            v1 = (current_transform * vec4(v1, 1.0f)).xyz();
            v2 = (current_transform * vec4(v2, 1.0f)).xyz();
            v3 = (current_transform * vec4(v3, 1.0f)).xyz();

            n1 = (trans_inv * vec4(n1, 1.0f)).xyz();
            n2 = (trans_inv * vec4(n2, 1.0f)).xyz();
            n3 = (trans_inv * vec4(n3, 1.0f)).xyz();

            gn = (trans_inv * vec4(gn, 1.0f)).xyz();

            auto& t = model_.primitives[first_primitive + i / 3];
            t = basic_triangle<3, float>(v1, v2 - v1, v3 - v1);
            t.prim_id = static_cast<unsigned>(first_primitive + i / 3);
            t.geom_id = current_geom_id;

            model_.shading_normals[first_shading_normal + i]     = n1;
            model_.shading_normals[first_shading_normal + i + 1] = n2;
            model_.shading_normals[first_shading_normal + i + 2] = n3;

            model_.geometric_normals[first_geometric_normal + i / 3] = gn;

            model_.tex_coords[first_tex_coord + i]     = tc1;
            model_.tex_coords[first_tex_coord + i + 1] = tc2;
            model_.tex_coords[first_tex_coord + i + 2] = tc3;

            model_.bbox.insert(v1);
            model_.bbox.insert(v2);
            model_.bbox.insert(v3);
        }

        node_visitor::apply(tm);
    }

    void apply(sphere& s)
    {
        // Matrix to transform normals
        mat4 trans_inv = inverse(transpose(current_transform));

        // Create triangles from sphere
        int resolution = 64;

        for (int i = 0; i < resolution; ++i)
        {
            int x1 = i;
            int x2 = (i + 1) % resolution;

            float theta1 = x1 * constants::two_pi<float>() / resolution - constants::pi_over_two<float>();
            float theta2 = x2 * constants::two_pi<float>() / resolution - constants::pi_over_two<float>();

            for (int j = 0; j < resolution / 2; ++j)
            {
                int y1 = j;
                int y2 = (j + 1) % resolution;

                float phi1 = y1 * constants::two_pi<float>() / resolution;
                float phi2 = y2 * constants::two_pi<float>() / resolution;

                vec3 v1(cos(theta1) * cos(phi1), sin(theta1), cos(theta1) * sin(phi1));
                vec3 v2(cos(theta2) * cos(phi1), sin(theta2), cos(theta2) * sin(phi1));
                vec3 v3(cos(theta2) * cos(phi2), sin(theta2), cos(theta2) * sin(phi2));
                vec3 v4(cos(theta1) * cos(phi2), sin(theta1), cos(theta1) * sin(phi2));

                vec3 n1 = v1;
                vec3 n2 = v2;
                vec3 n3 = v3;
                vec3 n4 = v4;

                vec3 gn1 = normalize(cross(v2 - v1, v3 - v1));
                vec3 gn2 = normalize(cross(v3 - v1, v4 - v1));

                vec2 tc1(i / static_cast<float>(resolution), j / static_cast<float>(resolution / 2));
                vec2 tc2((i + 1) / static_cast<float>(resolution), j / static_cast<float>(resolution / 2));
                vec2 tc3((i + 1) / static_cast<float>(resolution), (j + 1) / static_cast<float>(resolution / 2));
                vec2 tc4(i / static_cast<float>(resolution), (j + 1) / static_cast<float>(resolution / 2));

                v1 = (current_transform * vec4(v1, 1.0f)).xyz();
                v2 = (current_transform * vec4(v2, 1.0f)).xyz();
                v3 = (current_transform * vec4(v3, 1.0f)).xyz();
                v4 = (current_transform * vec4(v4, 1.0f)).xyz();

                n1 = (trans_inv * vec4(n1, 1.0f)).xyz();
                n2 = (trans_inv * vec4(n2, 1.0f)).xyz();
                n3 = (trans_inv * vec4(n3, 1.0f)).xyz();
                n4 = (trans_inv * vec4(n4, 1.0f)).xyz();

                if (i >= resolution / 2)
                {
                    n1 *= -1.0f;
                    n2 *= -1.0f;
                    n3 *= -1.0f;
                    n4 *= -1.0f;
                }

                gn1 = (trans_inv * vec4(gn1, 1.0f)).xyz();
                gn2 = (trans_inv * vec4(gn2, 1.0f)).xyz();

                basic_triangle<3, float> t1(v1, v2 - v1, v3 - v1);
                t1.prim_id = static_cast<unsigned>(model_.primitives.size());
                t1.geom_id = current_geom_id;
                model_.primitives.push_back(t1);

                basic_triangle<3, float> t2(v1, v3 - v1, v4 - v1);
                t2.prim_id = static_cast<unsigned>(model_.primitives.size());
                t2.geom_id = current_geom_id;
                model_.primitives.push_back(t2);

                model_.shading_normals.push_back(n1);
                model_.shading_normals.push_back(n2);
                model_.shading_normals.push_back(n3);

                model_.shading_normals.push_back(n1);
                model_.shading_normals.push_back(n3);
                model_.shading_normals.push_back(n4);

                model_.geometric_normals.push_back(gn1);
                model_.geometric_normals.push_back(gn2);

                model_.tex_coords.push_back(tc1);
                model_.tex_coords.push_back(tc2);
                model_.tex_coords.push_back(tc3);
                model_.tex_coords.push_back(tc4);

                model_.bbox.insert(v1);
                model_.bbox.insert(v2);
                model_.bbox.insert(v3);
                model_.bbox.insert(v4);
            }
        }

        node_visitor::apply(s);
    }

    // The model to add triangles to
    model& model_;

    // List of materials to derive geom_ids from
    std::vector<std::shared_ptr<sg::material>> materials;

    // Current transform w.r.t. transform nodes
    mat4 current_transform = mat4::identity();

    // Current geometry id w.r.t. material list
    unsigned current_geom_id = 0;
};

void flatten(model& mod, node& root)
{
    flatten_visitor visitor(mod);
    root.accept(visitor);

    for (auto& mat : visitor.materials)
    {
        model::material_type newmat = {};
        auto disney = std::dynamic_pointer_cast<sg::disney_material>(mat);
        assert(disney != nullptr);
        newmat.cd = disney->base_color.xyz();
        newmat.cs = vec3(0.0f);
        mod.materials.push_back(newmat); // TODO
    }
}

} // sg
} // visionaray
