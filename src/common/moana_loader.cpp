// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/utility/string_ref.hpp>
#include <boost/filesystem.hpp>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include "moana_loader.h"
#include "model.h"
#include "obj_grammar.h"
#include "sg.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Extract base path from json file
// As a heuristic, traverse the path from left to right until we find a folder json/
// The parent folder of the latter is the base path
//

inline boost::filesystem::path get_base_path(std::string const& filename)
{
    boost::filesystem::path p(filename);

    while (p.stem().string() != "json" || p.empty())
    {
        p = p.parent_path();
    }

    return p.parent_path();
}


//-------------------------------------------------------------------------------------------------
// Map obj indices to unsigned base-0 indices
//

template <typename Int>
inline Int remap_index(Int idx, Int size)
{
    return idx < 0 ? size + idx : idx - 1;
}

static void store_faces(
        std::shared_ptr<sg::triangle_mesh>& tm,
        vertex_vector const&                vertices,
        normal_vector const&                normals,
        face_vector const&                  faces
        )
{

    auto vertices_size = static_cast<int>(vertices.size());
    size_t last = 2;
    auto i1 = remap_index(faces[0].vertex_index, vertices_size);

    // simply construct new vertices for each obj face we encounter
    // ..too hard to keep track of v/vn combinations..
    while (last != faces.size())
    {
        // triangle indices
        auto i2 = remap_index(faces[last - 1].vertex_index, vertices_size);
        auto i3 = remap_index(faces[last].vertex_index, vertices_size);

        // no texture coordinates
        // ...

        // normal indices
        auto normals_size = static_cast<int>(normals.size());
        auto ni1 = remap_index(*faces[0].normal_index, normals_size);
        auto ni2 = remap_index(*faces[last - 1].normal_index, normals_size);
        auto ni3 = remap_index(*faces[last].normal_index, normals_size);

        tm->vertices.push_back({
            vertices[i1],
            normals[ni1],
            vec2(0.0f), // no tex coords
            vec4(0.0f)  // base color undefined
            });

        tm->vertices.push_back({
            vertices[i2],
            normals[ni2],
            vec2(0.0f), // no tex coords
            vec4(0.0f)  // base color undefined
            });

        tm->vertices.push_back({
            vertices[i3],
            normals[ni3],
            vec2(0.0f), // no tex coords
            vec4(0.0f)  // base color undefined
            });

        ++last;
    }
}

static void load_obj(
        std::string const& filename,
        std::map<std::string, std::shared_ptr<sg::disney_material>> const& materials,
        std::vector<std::shared_ptr<sg::node>>& objs
        )
{
    namespace qi = boost::spirit::qi;

    using boost::string_ref;

    boost::iostreams::mapped_file_source file(filename);

    obj_grammar grammar;

    string_ref text(file.data(), file.size());
    auto it = text.cbegin();

    // containers for parsing

    vertex_vector    vertices;
    tex_coord_vector tex_coords;
    normal_vector    normals;
    face_vector      faces;

    string_ref comment;
    string_ref mtl_file;
    string_ref mtl_name;

    std::shared_ptr<sg::surface_properties> surf = nullptr;
    std::shared_ptr<sg::triangle_mesh> tm = nullptr;

    while (it != text.cend())
    {
        faces.clear();
        if ( qi::phrase_parse(it, text.cend(), grammar.r_usemtl, qi::blank, mtl_name) )
        {
            objs.push_back(std::make_shared<sg::surface_properties>());
            objs.back()->add_child(std::make_shared<sg::triangle_mesh>());

            surf = std::dynamic_pointer_cast<sg::surface_properties>(objs.back());
            tm = std::dynamic_pointer_cast<sg::triangle_mesh>(surf->children().back());

            std::string usemtl = mtl_name.to_string();
            auto it = materials.find(usemtl);
            if (it != materials.end())
            {
                surf->material() = it->second;
            }
        }
        else if ( qi::phrase_parse(it, text.cend(), grammar.r_vertices, qi::blank, vertices) )
        {
        }
        else if ( qi::phrase_parse(it, text.cend(), grammar.r_normals, qi::blank, normals) )
        {
        }
        else if ( qi::phrase_parse(it, text.cend(), grammar.r_face, qi::blank, faces) )
        {
            store_faces(tm, vertices, normals, faces);
        }
        else if ( qi::phrase_parse(it, text.cend(), grammar.r_unhandled, qi::blank) )
        {
        }
        else
        {
            ++it;
        }
    }

    for (auto obj : objs)
    {
        auto surf = std::dynamic_pointer_cast<sg::surface_properties>(obj);

        for (auto c : surf->children())
        {
            auto tm = std::dynamic_pointer_cast<sg::triangle_mesh>(c);

            // Fill with 0,1,2,3,4,..
            tm->indices.resize(tm->vertices.size());
            std::iota(tm->indices.begin(), tm->indices.end(), 0);
        }
    }
}

static void load_instanced_primitive_json_file(
        boost::filesystem::path const& island_base_path,
        std::string const& filename,
        std::shared_ptr<sg::node> root,
        std::map<std::string, std::shared_ptr<sg::disney_material>>& materials
        )
{
    std::cout << "Load instanced primitive json file: " << (island_base_path / filename).string() << '\n';
    std::ifstream stream((island_base_path / filename).string());
    if (stream.fail())
    {
        std::cerr << "Cannot open " << filename << '\n';
        return;
    }

    rapidjson::IStreamWrapper isw(stream);
    rapidjson::Document doc;
    doc.ParseStream(isw);

    for (auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it)
    {
        auto entry = it->value.GetObject();

        // Instance geometry
        std::string obj_file = it->name.GetString();
        std::vector<std::shared_ptr<sg::node>> objs;
        load_obj((island_base_path / obj_file).string(), materials, objs);

        // Instance transforms
        auto entries = it->value.GetObject();
        for (auto it = entries.MemberBegin(); it != entries.MemberEnd(); ++it)
        {
            auto transform = std::make_shared<sg::transform>();
            int i = 0;
            rapidjson::Value const& tm = it->value;
            for (auto& item : tm.GetArray())
            {
                transform->matrix().data()[i++] = item.GetFloat();
                assert(i <= 16);
            }

            for (auto obj : objs)
            {
                transform->add_child(obj);
            }
            root->add_child(transform);
        }
    }
}

void load_material_file(
        std::string const& filename,
        std::map<std::string, std::shared_ptr<sg::disney_material>>& materials
        )
{
    std::cout << "Load material file: " << filename << '\n';
    std::ifstream stream(filename);
    if (stream.fail())
    {
        std::cerr << "Cannot open " << filename << '\n';
        return;
    }

    rapidjson::IStreamWrapper isw(stream);
    rapidjson::Document doc;
    doc.ParseStream(isw);

    for (auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it)
    {
        auto entry = it->value.GetObject();

        std::string material_name = it->name.GetString();

        std::shared_ptr<sg::disney_material> mat = std::make_shared<sg::disney_material>();

        int i = 0;
        rapidjson::Value const& bc = entry["baseColor"];
        for (auto& item : bc.GetArray())
        {
            mat->base_color[i++] = item.GetFloat();
            assert(i <= 4);
        }

        materials.insert({ material_name, mat });
    }
}

void load_moana(std::string const& filename, model& mod)
{
    std::cout << "Load moana file: " << filename << '\n';
    std::ifstream stream(filename);
    if (stream.fail())
    {
        std::cerr << "Cannot open " << filename << '\n';
        return;
    }

    boost::filesystem::path island_base_path = get_base_path(filename);

    if (island_base_path.empty())
    {
        std::cerr << "Cannot extract Moana Island Scene base path from " << filename << '\n';
        return;
    }

    rapidjson::IStreamWrapper isw(stream);
    rapidjson::Document doc;
    doc.ParseStream(isw);

    auto root = std::make_shared<sg::node>();


    // matFile
    std::string mat_file = doc["matFile"].GetString();
    std::map<std::string, std::shared_ptr<sg::disney_material>> materials;
    load_material_file((island_base_path / mat_file).string(), materials);

    auto base_transform = std::make_shared<sg::transform>();
    root->add_child(base_transform);

    // transformMatrix
    int i = 0;
    rapidjson::Value const& tm = doc["transformMatrix"];
    for (auto& item : tm.GetArray())
    {
        base_transform->matrix().data()[i++] = item.GetFloat();
        assert(i <= 16);
    }


    // geomObjFile
    if (doc.HasMember("geomObjFile"))
    {
        std::string geom_obj_file = doc["geomObjFile"].GetString();
        std::vector<std::shared_ptr<sg::node>> objs;
        load_obj((island_base_path / geom_obj_file).string(), materials, objs);
        for (auto obj : objs)
        {
            base_transform->add_child(obj);
        }
    }


    // instancedPrimitiveJsonFiles
    if (doc.HasMember("instancedPrimitiveJsonFiles"))
    {
        rapidjson::Value const& entries = doc["instancedPrimitiveJsonFiles"];
        for (auto it = entries.MemberBegin(); it != entries.MemberEnd(); ++it)
        {
            auto entry = it->value.GetObject();
            std::string type = entry["type"].GetString();

            if (type == "archive")
            {
                std::string inst_json_file = entry["jsonFile"].GetString();
                load_instanced_primitive_json_file(island_base_path, inst_json_file, base_transform, materials);
            }
        }
    }


    // instandedCopies (TODO: support copies with other geometry!)
    if (doc.HasMember("instancedCopies"))
    {
        rapidjson::Value const& entries = doc["instancedCopies"];
        for (auto it = entries.MemberBegin(); it != entries.MemberEnd(); ++it)
        {
            auto transform = std::make_shared<sg::transform>();
            root->add_child(transform);

            auto entry = it->value.GetObject();

            int i = 0;
            rapidjson::Value const& tm = entry["transformMatrix"];
            for (auto& item : tm.GetArray())
            {
                transform->matrix().data()[i++] = item.GetFloat();
                assert(i <= 16);
            }

            // Check if the copy has its own geomObjFile
            // (i.e. only texture and material are copied)
            if (entry.HasMember("geomObjFile"))
            {
                std::string geom_obj_file = entry["geomObjFile"].GetString();
                std::vector<std::shared_ptr<sg::node>> objs;
                load_obj((island_base_path / geom_obj_file).string(), materials, objs);
                for (auto obj : objs)
                {
                    transform->add_child(obj);
                }
            }
            else
            {
                // No. Rather the default case, add the top level
                // json file's geomObjFile content as an instance
                for (auto c : base_transform->children())
                {
                    transform->add_child(c);
                }
            }

            // Copies may also have their own instancedPrimitiveJsonFiles
            if (entry.HasMember("instancedPrimitiveJsonFiles"))
            {
                rapidjson::Value const& entries = entry["instancedPrimitiveJsonFiles"];
                for (auto it = entries.MemberBegin(); it != entries.MemberEnd(); ++it)
                {
                    auto entry = it->value.GetObject();

                    std::string type = entry["type"].GetString();

                    if (type == "archive")
                    {
                        std::string inst_json_file = entry["jsonFile"].GetString();
                        load_instanced_primitive_json_file(island_base_path, inst_json_file, transform, materials);
                    }
                }
            }
        }
    }


#if 0
    flatten(mod, *root);std::cout << mod.primitives.size() << '\n';
#else
    if (mod.scene_graph == nullptr)
    {
        mod.scene_graph = std::make_shared<sg::node>();
    }

    mod.scene_graph->add_child(root);
#endif
}

} // visionaray
