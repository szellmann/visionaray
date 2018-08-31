// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include "model.h"
#include "sg.h"

namespace visionaray
{
namespace sg
{

//-------------------------------------------------------------------------------------------------
// node
//

std::vector<node::node_pointer>& node::parents()
{
    return parents_;
}

std::vector<node::node_pointer> const& node::parents() const
{
    return parents_;
}

std::vector<node::node_pointer>& node::children()
{
    return children_;
}

std::vector<node::node_pointer> const& node::children() const
{
    return children_;
}

void node::add_child(node_pointer child)
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

void transform::set_matrix(mat4 matrix)
{
    matrix_ = matrix;
}

mat4 transform::get_matrix() const
{
    return matrix_;
}


//-------------------------------------------------------------------------------------------------
// triangle_mesh
//


//-------------------------------------------------------------------------------------------------
// sphere
//


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
            p->accept(*this);
        }
    }
}

void node_visitor::apply(transform& t)
{
    apply(static_cast<node&>(t));
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
// flatten
//

struct flatten_visitor : node_visitor
{
    flatten_visitor(model& mod)
        : model_(mod)
    {
    }

    void apply(transform& t)
    {
        mat4 prev = current_transform;

        current_transform = t.get_matrix() * current_transform;

        node_visitor::apply(t);

        current_transform = prev;
    }

    void apply(triangle_mesh& tm)
    {
        // Matrix to transform normals
        mat4 trans_inv = inverse(transpose(current_transform));

        assert(tm.indices.size() % 3 == 0);

        for (size_t i = 0; i < tm.indices.size(); i += 3)
        {
            vec3 v1 = tm.vertices[tm.indices[i]].pos;
            vec3 v2 = tm.vertices[tm.indices[i + 1]].pos;
            vec3 v3 = tm.vertices[tm.indices[i + 2]].pos;

            vec3 n1 = tm.vertices[tm.indices[i]].normal;
            vec3 n2 = tm.vertices[tm.indices[i + 1]].normal;
            vec3 n3 = tm.vertices[tm.indices[i + 2]].normal;

            vec3 gn = normalize(cross(v2 - v1, v3 - v1));

            v1 = (current_transform * vec4(v1, 1.0f)).xyz();
            v2 = (current_transform * vec4(v2, 1.0f)).xyz();
            v3 = (current_transform * vec4(v3, 1.0f)).xyz();

            n1 = (trans_inv * vec4(n1, 1.0f)).xyz();
            n2 = (trans_inv * vec4(n2, 1.0f)).xyz();
            n3 = (trans_inv * vec4(n3, 1.0f)).xyz();

            gn = (trans_inv * vec4(gn, 1.0f)).xyz();

            basic_triangle<3, float> t(v1, v2 - v1, v3 - v1);
            t.prim_id = static_cast<unsigned>(model_.primitives.size());
            t.geom_id = 0; // TODO
            model_.primitives.push_back(t);

            model_.shading_normals.push_back(n1);
            model_.shading_normals.push_back(n2);
            model_.shading_normals.push_back(n3);

            model_.geometric_normals.push_back(gn);

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
                t1.geom_id = 0; // TODO
                model_.primitives.push_back(t1);

                basic_triangle<3, float> t2(v1, v3 - v1, v4 - v1);
                t2.prim_id = static_cast<unsigned>(model_.primitives.size());
                t2.geom_id = 0; // TODO
                model_.primitives.push_back(t2);

                model_.shading_normals.push_back(n1);
                model_.shading_normals.push_back(n2);
                model_.shading_normals.push_back(n3);

                model_.shading_normals.push_back(n1);
                model_.shading_normals.push_back(n3);
                model_.shading_normals.push_back(n4);

                model_.geometric_normals.push_back(gn1);
                model_.geometric_normals.push_back(gn2);

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

    // Current transform w.r.t. transform nodes
    mat4 current_transform = mat4::identity();
};

void flatten(model& mod, node& root)
{
    flatten_visitor visitor(mod);
    root.accept(visitor);
    mod.materials.push_back({}); // TODO
}

} // sg
} // visionaray
