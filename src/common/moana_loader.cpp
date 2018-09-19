// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

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

#if VSNRAY_COMMON_HAVE_PTEX
#include <Ptexture.h>
#endif

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include "moana_loader.h"
#include "model.h"
#include "obj_grammar.h"
#include "sg.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Remove first element from path
//

boost::filesystem::path remove_first(boost::filesystem::path const& p)
{
    auto parent_path = p.parent_path();
    if (parent_path.empty())
    {
        return boost::filesystem::path();
    }
    else
    {
        return remove_first(parent_path) / p.filename();
    }
}

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
        face_vector const&                  faces,
        int                                 face_id
        )
{
    // Ptex is a per-face texture format. We convert into one
    // with UVs for compatibility with Visionaray: assign simple
    // [0..1/0..1] UVs for quads
    vec2 tex_coords[] = {
        vec2(0.0f, 0.0f),
        vec2(1.0f, 0.0f),
        vec2(1.0f, 1.0f),
        vec2(0.0f, 1.0f)
        };

    auto vertices_size = static_cast<int>(vertices.size());
    size_t last = 2;
    auto i1 = remap_index(faces[0].vertex_index, vertices_size);
    auto tc1 = tex_coords[0];

    // simply construct new vertices for each obj face we encounter
    // ..too hard to keep track of v/vn combinations..
    while (last != faces.size())
    {
        // triangle indices
        auto i2 = remap_index(faces[last - 1].vertex_index, vertices_size);
        auto i3 = remap_index(faces[last].vertex_index, vertices_size);

        // texture coordinates
        auto tc2 = tex_coords[last - 1];
        auto tc3 = tex_coords[last];

        // normal indices
        auto normals_size = static_cast<int>(normals.size());
        auto ni1 = remap_index(*faces[0].normal_index, normals_size);
        auto ni2 = remap_index(*faces[last - 1].normal_index, normals_size);
        auto ni3 = remap_index(*faces[last].normal_index, normals_size);

        tm->vertices.push_back({
            vertices[i1],
            normals[ni1],
            tc1,
            vec4(0.0f),  // base color undefined
            face_id
            });

        tm->vertices.push_back({
            vertices[i2],
            normals[ni2],
            tc2,
            vec4(0.0f),  // base color undefined
            face_id
            });

        tm->vertices.push_back({
            vertices[i3],
            normals[ni3],
            tc3,
            vec4(0.0f),  // base color undefined
            face_id
            });

        ++last;
    }

    // Should all be quad faces for subdiv
    assert(last == 4);
}


//-------------------------------------------------------------------------------------------------
// Load ptx texture
//

static std::shared_ptr<sg::ptex_texture> load_texture(
        boost::filesystem::path const& texture_base_path,
        std::string const& filename
        )
{
    auto fn = (texture_base_path / filename).string();

    if (!boost::filesystem::exists(fn))
    {
        return nullptr;
    }

#if VSNRAY_COMMON_HAVE_PTEX

    Ptex::String error = "";
    PtexPtr<PtexTexture> tex(PtexTexture::open(fn.c_str(), error));

    if (tex == nullptr)
    {
        std::cerr << "Error: " << error << '\n';
        return nullptr;
    }
    else
    {
        // sg::ptex_texture's ctor transfers ownership of texptr
        auto tex_node = std::make_shared<sg::ptex_texture>(tex);

        return tex_node;
    }
#else

    std::cerr << "Warning: not compiled with Ptex support\n";

    return nullptr;
#endif
}


//-------------------------------------------------------------------------------------------------
// Load obj file
//

static void load_obj(
        boost::filesystem::path const& island_base_path,
        std::string const& filename,
        std::map<std::string, std::shared_ptr<sg::disney_material>> const& materials,
        std::map<std::string, std::shared_ptr<sg::ptex_texture>>& textures,
        std::vector<std::shared_ptr<sg::node>>& objs
        )
{
    namespace qi = boost::spirit::qi;

    using boost::string_ref;

    boost::iostreams::mapped_file_source file((island_base_path / filename).string());

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
    string_ref group_name;
    string_ref mtl_name;

    std::shared_ptr<sg::surface_properties> surf = nullptr;
    std::shared_ptr<sg::triangle_mesh> tm = nullptr;

    // Face ID for ptex
    int face_id = 0;

    while (it != text.cend())
    {
        faces.clear();
        if ( qi::phrase_parse(it, text.cend(), grammar.r_g, qi::blank, group_name) )
        {
            if (group_name != "default")
            {
                objs.push_back(std::make_shared<sg::surface_properties>());
                objs.back()->add_child(std::make_shared<sg::triangle_mesh>());

                surf = std::dynamic_pointer_cast<sg::surface_properties>(objs.back());
                tm = std::dynamic_pointer_cast<sg::triangle_mesh>(surf->children().back());

                surf->name() = std::string(group_name.begin(), group_name.length());

                std::string group = group_name.to_string();
                auto it = textures.find(group);
                if (it != textures.end())
                {
                    surf->add_texture(std::static_pointer_cast<sg::texture>(it->second));
                }
                else
                {
                    // Extract element base name from obj file path
                    boost::filesystem::path obj_path(filename);
                    // Remove obj file name
                    boost::filesystem::path texture_base_path = obj_path.parent_path();
                    // If archive, remove archives/
                    if (texture_base_path.filename().string() == "archives")
                    {
                        texture_base_path = texture_base_path.parent_path();
                    }
                    // Remove leading obj/
                    texture_base_path = remove_first(texture_base_path);
                    // Combine with base path, add textures/ and Color/
                    texture_base_path = island_base_path / "textures" / texture_base_path / "Color";

                    auto tex = load_texture(texture_base_path, group + ".ptx");

                    if (tex != nullptr)
                    {
                        textures.insert(std::make_pair(group, tex));
                        surf->add_texture(std::static_pointer_cast<sg::texture>(tex));
                    }
                }
            }
        }
        else if ( qi::phrase_parse(it, text.cend(), grammar.r_usemtl, qi::blank, mtl_name) )
        {
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
            store_faces(tm, vertices, normals, faces, face_id++);
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

        if (surf->textures().size() == 0)
        {
            auto it = textures.find("null");
            surf->add_texture(it->second);
        }

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
        std::map<std::string, std::shared_ptr<sg::disney_material>>& materials,
        std::map<std::string, std::shared_ptr<sg::ptex_texture>>& textures
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
        load_obj(island_base_path, obj_file, materials, textures, objs);

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
        boost::filesystem::path const& island_base_path,
        std::string const& filename,
        std::map<std::string, std::shared_ptr<sg::disney_material>>& materials
        )
{
    std::string fn = (island_base_path / filename).string();

    std::cout << "Load material file: " << fn << '\n';
    std::ifstream stream(fn);
    if (stream.fail())
    {
        std::cerr << "Cannot open " << fn << '\n';
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

        if (entry.HasMember("baseColor"))
        {
            int i = 0;
            rapidjson::Value const& bc = entry["baseColor"];
            for (auto& item : bc.GetArray())
            {
                mat->base_color[i++] = item.GetFloat();
                assert(i <= 4);
            }
        }

        if (entry.HasMember("colorMap"))
        {
            std::string path = entry["colorMap"].GetString();

            if (!path.empty())
            {
            }
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
    load_material_file(island_base_path, mat_file, materials);

    // Textures, init with one empty texture
    std::map<std::string, std::shared_ptr<sg::ptex_texture>> textures;
    PtexPtr<PtexTexture> dummy(nullptr);
    textures.insert(std::make_pair("null", std::make_shared<sg::ptex_texture>(dummy)));

    auto base_transform = std::make_shared<sg::transform>();
    root->add_child(base_transform);

    // transformMatrix
    int i = 0;
    rapidjson::Value const& tm = doc["transformMatrix"];
    if (tm.IsArray())
    {
        for (auto& item : tm.GetArray())
        {
            base_transform->matrix().data()[i++] = item.GetFloat();
            assert(i <= 16);
        }
    }


    // geomObjFile
    if (doc.HasMember("geomObjFile"))
    {
        std::string geom_obj_file = doc["geomObjFile"].GetString();
        std::vector<std::shared_ptr<sg::node>> objs;
        load_obj(island_base_path, geom_obj_file, materials, textures, objs);
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
                load_instanced_primitive_json_file(
                        island_base_path,
                        inst_json_file,
                        base_transform,
                        materials,
                        textures
                        );
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
                load_obj(island_base_path, geom_obj_file, materials, textures, objs);
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
                        load_instanced_primitive_json_file(
                                island_base_path,
                                inst_json_file,
                                transform,
                                materials,
                                textures
                                );
                    }
                }
            }
        }
    }

    mod.tex_format = model::Ptex;

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
