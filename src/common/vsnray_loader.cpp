// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <unordered_map>

#include <boost/filesystem.hpp>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

#include <visionaray/math/constants.h>
#include <visionaray/math/forward.h>
#include <visionaray/math/unorm.h>
#include <visionaray/math/vector.h>
#include <visionaray/texture/texture.h>

#include "cfile.h"
#include "model.h"
#include "sg.h"
#include "vsnray_loader.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// .vsnray parser
//

class vsnray_parser
{
public:

    vsnray_parser(std::string filename)
        : filename_(filename)
    {
    }

    void parse_children(std::shared_ptr<sg::node> parent, rapidjson::Value const& entries);

    template <typename Object>
    std::shared_ptr<sg::node> parse_node(Object const& obj);

    template <typename Object>
    std::shared_ptr<sg::node> parse_camera(Object const& obj);

    template <typename Object>
    std::shared_ptr<sg::node> parse_include(Object const& obj);

    template <typename Object>
    std::shared_ptr<sg::node> parse_point_light(Object const& obj);

    template <typename Object>
    std::shared_ptr<sg::node> parse_reference(Object const& obj);

    template <typename Object>
    std::shared_ptr<sg::node> parse_transform(Object const& obj);

    template <typename Object>
    std::shared_ptr<sg::node> parse_surface_properties(Object const& obj);

    template <typename Object>
    std::shared_ptr<sg::node> parse_triangle_mesh(Object const& obj);

    template <typename Object>
    std::shared_ptr<sg::node> parse_indexed_triangle_mesh(Object const& obj);

private:

    std::string filename_;

};


//-------------------------------------------------------------------------------------------------
// Parse nodes
//

void vsnray_parser::parse_children(std::shared_ptr<sg::node> parent, rapidjson::Value const& entries)
{
    if (!entries.IsArray())
    {
        throw std::runtime_error("");
    }

    parent->children().resize(entries.Capacity());

    size_t i = 0;
    for (auto const& c : entries.GetArray())
    {
        auto const& obj = c.GetObject();

        parent->children().at(i++) = parse_node(obj);
    }

    if (i != entries.Capacity())
    {
        throw std::runtime_error("");
    }
}

template <typename Object>
std::shared_ptr<sg::node> vsnray_parser::parse_node(Object const& obj)
{
    std::shared_ptr<sg::node> result = nullptr;

    if (obj.HasMember("type"))
    {
        // Parse individual node types
        auto const& type_string = obj["type"];
        if (strncmp(type_string.GetString(), "node", 4) == 0)
        {
            // Empty node, (may still contain children, e.g. root)
            result = std::make_shared<sg::node>();
        }
        else if (strncmp(type_string.GetString(), "camera", 6) == 0)
        {
            result = parse_camera(obj);
        }
        else if (strncmp(type_string.GetString(), "include", 6) == 0)
        {
            result = parse_include(obj);
        }
        else if (strncmp(type_string.GetString(), "point_light", 11) == 0)
        {
            result = parse_point_light(obj);
        }
        else if (strncmp(type_string.GetString(), "reference", 9) == 0)
        {
            result = parse_reference(obj);
        }
        else if (strncmp(type_string.GetString(), "transform", 9) == 0)
        {
            result = parse_transform(obj);
        }
        else if (strncmp(type_string.GetString(), "surface_properties", 18) == 0)
        {
            result = parse_surface_properties(obj);
        }
        else if (strncmp(type_string.GetString(), "triangle_mesh", 13) == 0)
        {
            result = parse_triangle_mesh(obj);
        }
        else if (strncmp(type_string.GetString(), "indexed_triangle_mesh", 21) == 0)
        {
            result = parse_indexed_triangle_mesh(obj);
        }
        else
        {
            throw std::runtime_error("");
        }

        // Parse common node properties
        if (obj.HasMember("name"))
        {
            assert(result != nullptr);

            rapidjson::Value const& name = obj["name"];
            result->name() = name.GetString();
        }

        if (obj.HasMember("children"))
        {
            assert(result != nullptr);

            rapidjson::Value const& children = obj["children"];
            parse_children(result, children);
        }
    }
    else
    {
        throw std::runtime_error("");
    }

    return result;
}

template <typename Object>
std::shared_ptr<sg::node> vsnray_parser::parse_camera(Object const& obj)
{
    auto cam = std::make_shared<sg::camera>();

    vec3 eye(0.0f);
    if (obj.HasMember("eye"))
    {
        auto const& cam_eye = obj["eye"];

        int i = 0;
        for (auto const& item : cam_eye.GetArray())
        {
            eye[i++] = item.GetFloat();
        }

        if (i != 3)
        {
            throw std::runtime_error("");
        }
    }

    vec3 center(0.0f);
    if (obj.HasMember("center"))
    {
        auto const& cam_center = obj["center"];

        int i = 0;
        for (auto const& item : cam_center.GetArray())
        {
            center[i++] = item.GetFloat();
        }

        if (i != 3)
        {
            throw std::runtime_error("");
        }
    }

    vec3 up(0.0f);
    if (obj.HasMember("up"))
    {
        auto const& cam_up = obj["up"];

        int i = 0;
        for (auto const& item : cam_up.GetArray())
        {
            up[i++] = item.GetFloat();
        }

        if (i != 3)
        {
            throw std::runtime_error("");
        }
    }

    float fovy = 45.0f;
    if (obj.HasMember("fovy"))
    {
        fovy = obj["fovy"].GetFloat();
    }

    float znear = 0.001f;
    if (obj.HasMember("znear"))
    {
        znear = obj["znear"].GetFloat();
    }

    float zfar = 1000.0f;
    if (obj.HasMember("zfar"))
    {
        zfar = obj["zfar"].GetFloat();
    }

    recti viewport(0, 0, 0, 0);
    if (obj.HasMember("viewport"))
    {
        auto const& cam_viewport = obj["viewport"];

        int i = 0;
        for (auto const& item : cam_viewport.GetArray())
        {
            viewport.data()[i++] = item.GetInt();
        }

        if (i != 4)
        {
            throw std::runtime_error("");
        }
    }

    float lens_radius = 0.1f;
    if (obj.HasMember("lens_radius"))
    {
        lens_radius = obj["lens_radius"].GetFloat();
    }

    float focal_distance = 10.0f;
    if (obj.HasMember("focal_distance"))
    {
        focal_distance = obj["focal_distance"].GetFloat();
    }

    float aspect = viewport.w > 0 && viewport.h > 0
                 ? viewport.w / static_cast<float>(viewport.h)
                 : 1;

    cam->perspective(fovy * constants::degrees_to_radians<float>(), aspect, znear, zfar);
    if (viewport.w > 0 && viewport.h > 0)
    {
        cam->set_viewport(viewport);
    }
    cam->set_lens_radius(lens_radius);
    cam->set_focal_distance(focal_distance);
    cam->look_at(eye, center, up);

    return cam;
}

template <typename Object>
std::shared_ptr<sg::node> vsnray_parser::parse_include(Object const& obj)
{
    auto inc = std::make_shared<sg::node>();

    if (obj.HasMember("path"))
    {
        std::string path_string(obj["path"].GetString());

        boost::filesystem::path p(path_string);

        if (!p.is_absolute())
        {
            // Extract base path
            boost::filesystem::path bp(filename_);
            bp = bp.parent_path();

            // Append path to base path
            p = bp / p;

            path_string = p.string();
        }

        model mod;
        if (mod.load(path_string))
        {
            if (mod.scene_graph == nullptr)
            {
                std::unordered_map<std::string, std::shared_ptr<sg::texture2d<vector<4, unorm<8>>>>> texture_map;

                for (auto it = mod.texture_map.begin(); it != mod.texture_map.end(); ++it)
                {
                    auto tex = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>();
                    tex->name() = it->first;
                    tex->resize(it->second.width(), it->second.height());
                    tex->reset(it->second.data());
                    tex->set_filter_mode(it->second.get_filter_mode());
                    tex->set_address_mode(it->second.get_address_mode());

                    texture_map.insert(std::make_pair(it->first, tex));
                }

                if (mod.primitives.size() > 0)
                {
                    // Vertices (disassemble triangles..)
                    for (auto tri : mod.primitives)
                    {
                        if (tri.geom_id >= inc->children().size())
                        {
                            auto props = std::make_shared<sg::surface_properties>();

                            // Add material
                            auto obj = std::make_shared<sg::obj_material>();
                            obj->ca = mod.materials[tri.geom_id].ca;
                            obj->cd = mod.materials[tri.geom_id].cd;
                            obj->cs = mod.materials[tri.geom_id].cs;
                            obj->ce = mod.materials[tri.geom_id].ce;
                            obj->cr = mod.materials[tri.geom_id].cr;
                            obj->ior = mod.materials[tri.geom_id].ior;
                            obj->absorption = mod.materials[tri.geom_id].absorption;
                            obj->transmission = mod.materials[tri.geom_id].transmission;
                            obj->specular_exp = mod.materials[tri.geom_id].specular_exp;
                            obj->illum = mod.materials[tri.geom_id].illum;
                            props->material() = obj;

                            bool insert_dummy = false;

                            if (tri.geom_id < mod.textures.size())
                            {
                                // Find texture in texture_map
                                bool found = false;
                                for (auto it = mod.texture_map.begin(); it != mod.texture_map.end(); ++it)
                                {
                                    auto ref = texture_ref<vector<4, unorm<8>>, 2>(it->second);

                                    if (ref.data() == mod.textures[tri.geom_id].data())
                                    {
                                        std::string name = it->first;
                                        // Find in local texture map
                                        auto res = texture_map.find(name);
                                        if (res != texture_map.end())
                                        {
                                            props->add_texture(res->second);
                                            found = true;
                                            break;
                                        }
                                    }
                                }

                                if (!found)
                                {
                                    insert_dummy = true;
                                }
                            }
                            else
                            {
                                insert_dummy = true;
                            }

                            if (insert_dummy)
                            {
                                // Add a dummy texture
                                vector<4, unorm<8>> dummy_texel(1.0f, 1.0f, 1.0f, 1.0f);
                                auto tex = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>();
                                tex->resize(1, 1);
                                tex->set_address_mode(Wrap);
                                tex->set_filter_mode(Nearest);
                                tex->reset(&dummy_texel);
                                props->add_texture(tex);
                            }

                            // Add to scene graph
                            props->add_child(std::make_shared<sg::triangle_mesh>());
                            inc->add_child(props);
                        }

                        auto mesh = std::dynamic_pointer_cast<sg::triangle_mesh>(
                                inc->children()[tri.geom_id]->children()[0]
                                );

                        vec3 verts[3] = {
                            tri.v1,
                            tri.v1 + tri.e1,
                            tri.v1 + tri.e2
                            };
                        mesh->vertices.insert(mesh->vertices.end(), verts, verts + 3);

                        if (mod.shading_normals.size() > tri.prim_id * 3 + 3)
                        {
                            for (int i = 0; i < 3; ++i)
                            {
                                mesh->normals.push_back(mod.shading_normals[tri.prim_id * 3 + i]);
                            }
                        }
                        else
                        {
                            for (int i = 0; i < 3; ++i)
                            {
                                mesh->normals.push_back(normalize(cross(tri.e1, tri.e2)));
                            }
                        }

                        if (mod.tex_coords.size() >= tri.prim_id * 3 + 3)
                        {
                            for (int i = 0; i < 3; ++i)
                            {
                                mesh->tex_coords.push_back(mod.tex_coords[tri.prim_id * 3 + i]);
                            }
                        }
                        else
                        {
                            for (int i = 0; i < 3; ++i)
                            {
                                mesh->tex_coords.push_back(vec2(0.0f, 0.0f));
                            }
                        }

                        if (mod.colors.size() >= tri.prim_id * 3 + 3)
                        {
                            for (int i = 0; i < 3; ++i)
                            {
                                mesh->colors.push_back(vector<3, unorm<8>>(mod.colors[tri.prim_id * 3 + i]));
                            }
                        }
                        else
                        {
                            for (int i = 0; i < 3; ++i)
                            {
                                mesh->colors.push_back(vector<3, unorm<8>>(1.0f, 1.0f, 1.0f));
                            }
                        }
                    }
                }
                else
                {
                    throw std::runtime_error("");
                }
            }
            else
            {
                // TODO: don't allow circular references..
                inc = mod.scene_graph;
            }
        }
        else
        {
            throw std::runtime_error("");
        }
    }
    else
    {
        throw std::runtime_error("");
    }

    return inc;
}

template <typename Object>
std::shared_ptr<sg::node> vsnray_parser::parse_point_light(Object const& obj)
{
    auto light = std::make_shared<sg::point_light>();

    vec3 cl(1.0f);
    if (obj.HasMember("cl"))
    {
        auto const& color = obj["cl"];

        int i = 0;
        for (auto const& item : color.GetArray())
        {
            cl[i++] = item.GetFloat();
        }

        if (i != 3)
        {
            throw std::runtime_error("");
        }
    }

    float kl = 1.0f;
    if (obj.HasMember("kl"))
    {
        kl = obj["kl"].GetFloat();
    }

    vec3 position(0.0f);
    if (obj.HasMember("position"))
    {
        auto const& pos = obj["position"];

        int i = 0;
        for (auto const& item : pos.GetArray())
        {
            position[i++] = item.GetFloat();
        }

        if (i != 3)
        {
            throw std::runtime_error("");
        }
    }

    float constant_attenuation = 1.0f;
    if (obj.HasMember("constant_attenuation"))
    {
        constant_attenuation = obj["constant_attenuation"].GetFloat();
    }

    float linear_attenuation = 0.0f;
    if (obj.HasMember("linear_attenuation"))
    {
        linear_attenuation = obj["linear_attenuation"].GetFloat();
    }

    float quadratic_attenuation = 0.0f;
    if (obj.HasMember("quadratic_attenuation"))
    {
        quadratic_attenuation = obj["quadratic_attenuation"].GetFloat();
    }

    light->set_cl(cl);
    light->set_kl(kl);
    light->set_position(position);
    light->set_constant_attenuation(constant_attenuation);
    light->set_linear_attenuation(linear_attenuation);
    light->set_quadratic_attenuation(quadratic_attenuation);

    return light;
}

template <typename Object>
std::shared_ptr<sg::node> vsnray_parser::parse_reference(Object const& obj)
{
    return std::make_shared<sg::node>();
}

template <typename Object>
std::shared_ptr<sg::node> vsnray_parser::parse_transform(Object const& obj)
{
    auto transform = std::make_shared<sg::transform>();

    if (obj.HasMember("matrix"))
    {
        auto const& mat = obj["matrix"];

        int i = 0;
        for (auto const& item : mat.GetArray())
        {
            transform->matrix().data()[i++] = item.GetFloat();
            assert(i <= 16);
        }
    }

    return transform;
}

template <typename Object>
std::shared_ptr<sg::node> vsnray_parser::parse_surface_properties(Object const& obj)
{
    auto props = std::make_shared<sg::surface_properties>();

    if (obj.HasMember("material"))
    {
        auto const& mat = obj["material"];

        if (mat.HasMember("type"))
        {
            auto const& type_string = mat["type"];
            if (strncmp(type_string.GetString(), "obj", 3) == 0)
            {
                auto obj = std::make_shared<sg::obj_material>();

                if (mat.HasMember("ca"))
                {
                    auto const& ca = mat["ca"];

                    vec3 clr;
                    int i = 0;
                    for (auto const& item : ca.GetArray())
                    {
                        clr[i++] = item.GetFloat();
                    }

                    if (i != 3)
                    {
                        throw std::runtime_error("");
                    }

                    obj->ca = clr;
                }

                if (mat.HasMember("cd"))
                {
                    auto const& cd = mat["cd"];

                    vec3 clr;
                    int i = 0;
                    for (auto const& item : cd.GetArray())
                    {
                        clr[i++] = item.GetFloat();
                    }

                    if (i != 3)
                    {
                        throw std::runtime_error("");
                    }

                    obj->cd = clr;
                }

                if (mat.HasMember("cs"))
                {
                    auto const& cs = mat["cs"];

                    vec3 clr;
                    int i = 0;
                    for (auto const& item : cs.GetArray())
                    {
                        clr[i++] = item.GetFloat();
                    }

                    if (i != 3)
                    {
                        throw std::runtime_error("");
                    }

                    obj->cs = clr;
                }

                if (mat.HasMember("ce"))
                {
                    auto const& ce = mat["ce"];

                    vec3 clr;
                    int i = 0;
                    for (auto const& item : ce.GetArray())
                    {
                        clr[i++] = item.GetFloat();
                    }

                    if (i != 3)
                    {
                        throw std::runtime_error("");
                    }

                    obj->ce = clr;
                }

                props->material() = obj;
            }
            else
            {
                throw std::runtime_error("");
            }
        }
        else
        {
            throw std::runtime_error("");
        }
    }
    else
    {
        // Set default material (wavefront obj)
        auto obj = std::make_shared<sg::obj_material>();
        props->material() = obj;
    }

    if (obj.HasMember("diffuse"))
    {
        // TODO: load from file
#if 1
        vector<4, unorm<8>> dummy_texel(1.0f, 1.0f, 1.0f, 1.0f);
        auto tex = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>();
        tex->resize(1, 1);
        tex->set_address_mode(Wrap);
        tex->set_filter_mode(Nearest);
        tex->reset(&dummy_texel);

        props->add_texture(tex);
#endif
    }
    else
    {
        // Set a dummy texture
        vector<4, unorm<8>> dummy_texel(1.0f, 1.0f, 1.0f, 1.0f);
        auto tex = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>();
        tex->resize(1, 1);
        tex->set_address_mode(Wrap);
        tex->set_filter_mode(Nearest);
        tex->reset(&dummy_texel);

        props->add_texture(tex);
    }

    return props;
}

template <typename Object>
std::shared_ptr<sg::node> vsnray_parser::parse_triangle_mesh(Object const& obj)
{
    auto mesh = std::make_shared<sg::triangle_mesh>();

    if (obj.HasMember("vertices"))
    {
        auto const& verts = obj["vertices"];

        vec3 v;
        int i = 0;
        for (auto const& item : verts.GetArray())
        {
            v[i++ % 3] = item.GetFloat();

            if (i % 3 == 0)
            {
                mesh->vertices.emplace_back(v);
            }
        }
    }

    if (obj.HasMember("normals"))
    {
        auto const& normals = obj["normals"];

        vec3 n;
        int i = 0;
        for (auto const& item : normals.GetArray())
        {
            n[i++ % 3] = item.GetFloat();

            if (i % 3 == 0)
            {
                mesh->normals.emplace_back(n);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < mesh->vertices.size(); i += 3)
        {
            vec3 v1 = mesh->vertices[i];
            vec3 v2 = mesh->vertices[i + 1];
            vec3 v3 = mesh->vertices[i + 2];

            vec3 gn = normalize(cross(v2 - v1, v3 - v1));

            mesh->normals.emplace_back(gn);
            mesh->normals.emplace_back(gn);
            mesh->normals.emplace_back(gn);
        }
    }

    if (obj.HasMember("tex_coords"))
    {
        auto const& tex_coords = obj["tex_coords"];

        vec3 tc;
        int i = 0;
        for (auto const& item : tex_coords.GetArray())
        {
            tc[i++ % 2] = item.GetFloat();

            if (i % 2 == 0)
            {
                mesh->tex_coords.emplace_back(tc);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < mesh->vertices.size(); ++i)
        {
            mesh->tex_coords.emplace_back(0.0f, 0.0f);
        }
    }

    if (obj.HasMember("colors"))
    {
        auto const& colors = obj["colors"];

        vector<3, unorm<8>> c;
        int i = 0;
        for (auto const& item : colors.GetArray())
        {
            c[i++ % 3] = item.GetFloat();

            if (i % 3 == 0)
            {
                mesh->colors.emplace_back(c);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < mesh->vertices.size(); ++i)
        {
            mesh->colors.emplace_back(1.0f);
        }
    }

    return mesh;
}

template <typename Object>
std::shared_ptr<sg::node> vsnray_parser::parse_indexed_triangle_mesh(Object const& obj)
{
    auto mesh = std::make_shared<sg::indexed_triangle_mesh>();

    if (obj.HasMember("indices"))
    {
        auto const& indices = obj["indices"];

        for (auto const& item : indices.GetArray())
        {
            mesh->indices.push_back(item.GetInt());
        }
    }

    if (obj.HasMember("vertices"))
    {
        auto const& verts = obj["vertices"];

        vec3 v;
        int i = 0;
        for (auto const& item : verts.GetArray())
        {
            v[i++ % 3] = item.GetFloat();

            if (i % 3 == 0)
            {
                mesh->vertices.emplace_back(v);
            }
        }
    }

    if (obj.HasMember("normals"))
    {
        auto const& normals = obj["normals"];

        vec3 n;
        int i = 0;
        for (auto const& item : normals.GetArray())
        {
            n[i++ % 3] = item.GetFloat();

            if (i % 3 == 0)
            {
                mesh->normals.emplace_back(n);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < mesh->vertices.size(); i += 3)
        {
            vec3 v1 = mesh->vertices[i];
            vec3 v2 = mesh->vertices[i + 1];
            vec3 v3 = mesh->vertices[i + 2];

            vec3 gn = normalize(cross(v2 - v1, v3 - v1));

            mesh->normals.emplace_back(gn);
            mesh->normals.emplace_back(gn);
            mesh->normals.emplace_back(gn);
        }
    }

    if (obj.HasMember("tex_coords"))
    {
        auto const& tex_coords = obj["tex_coords"];

        vec3 tc;
        int i = 0;
        for (auto const& item : tex_coords.GetArray())
        {
            tc[i++ % 2] = item.GetFloat();

            if (i % 2 == 0)
            {
                mesh->tex_coords.emplace_back(tc);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < mesh->vertices.size(); ++i)
        {
            mesh->tex_coords.emplace_back(0.0f, 0.0f);
        }
    }

    if (obj.HasMember("colors"))
    {
        auto const& colors = obj["colors"];

        vector<3, unorm<8>> c;
        int i = 0;
        for (auto const& item : colors.GetArray())
        {
            c[i++ % 3] = item.GetFloat();

            if (i % 3 == 0)
            {
                mesh->colors.emplace_back(c);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < mesh->vertices.size(); ++i)
        {
            mesh->colors.emplace_back(1.0f);
        }
    }

    return mesh;
}


//-------------------------------------------------------------------------------------------------
// Interface
//

void load_vsnray(std::string const& filename, model& mod)
{
    std::vector<std::string> filenames(1);

    filenames[0] = filename;

    load_vsnray(filenames, mod);
}

void load_vsnray(std::vector<std::string> const& filenames, model& mod)
{
    auto root = std::make_shared<sg::node>();

    for (auto filename : filenames)
    {
        cfile file(filename, "r");
        if (!file.good())
        {
            std::cerr << "Cannot open " << filename << '\n';
            return;
        }

        char buffer[65536];
        rapidjson::FileReadStream frs(file.get(), buffer, sizeof(buffer));
        rapidjson::Document doc;
        doc.ParseStream(frs);

        if (doc.IsObject())
        {
            vsnray_parser parser(filename);
            root = parser.parse_node(doc.GetObject());
        }
        else
        {
            throw std::runtime_error("");
        }
    }

    if (mod.scene_graph == nullptr)
    {
        mod.scene_graph = std::make_shared<sg::node>();
    }

    mod.scene_graph->add_child(root);
}

} // visionaray
