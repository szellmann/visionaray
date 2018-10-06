// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
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
#include <rapidjson/filereadstream.h>

#include <visionaray/math/forward.h>
#include <visionaray/math/rectangle.h>
#include <visionaray/math/vector.h>

#include "cfile.h"
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

        // no texture coordinates but face ids

        // normal indices
        auto normals_size = static_cast<int>(normals.size());
        auto ni1 = remap_index(*faces[0].normal_index, normals_size);
        auto ni2 = remap_index(*faces[last - 1].normal_index, normals_size);
        auto ni3 = remap_index(*faces[last].normal_index, normals_size);

        tm->vertices.emplace_back(vertices[i1]);
        tm->vertices.emplace_back(vertices[i2]);
        tm->vertices.emplace_back(vertices[i3]);

        tm->normals.emplace_back(normals[ni1]);
        tm->normals.emplace_back(normals[ni2]);
        tm->normals.emplace_back(normals[ni3]);

        tm->face_ids.emplace_back(face_id);

        ++last;
        face_id = ~face_id; // indicates 2nd triangle in quad
    }

    // Usually, but not always last == 4 (quads)
    // osOcean e.g. contains triangles
}


//-------------------------------------------------------------------------------------------------
// Load ptx texture
//

static std::shared_ptr<sg::texture> load_texture(
        boost::filesystem::path const& texture_base_path,
        std::string const& filename,
#if VSNRAY_COMMON_HAVE_PTEX
        std::shared_ptr<PtexPtr<PtexCache>> texture_cache
#else
        void* texture_cache
#endif
        )
{
    auto fn = (texture_base_path / filename).string();

    if (!boost::filesystem::exists(fn))
    {
        return nullptr;
    }

#if VSNRAY_COMMON_HAVE_PTEX

    Ptex::String error = "";
    PtexPtr<PtexTexture> tex(texture_cache->get()->get(fn.c_str(), error));

    if (tex == nullptr)
    {
        std::cerr << "Error: " << error << '\n';
        return nullptr;
    }
    else
    {
        // sg::ptex_texture's ctor transfers ownership of texptr
        auto tex_node = std::make_shared<sg::ptex_texture>(fn.c_str(), texture_cache);

        return tex_node;
    }
#else

    std::cerr << "Warning: not compiled with Ptex support\n";

    return nullptr;
#endif
}


//-------------------------------------------------------------------------------------------------
// Load camera from json file
//

static void load_camera(std::string const& filename, std::shared_ptr<sg::node> root)
{
    std::cout << "Load camera json file: " << filename << '\n';
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

    auto cam = std::make_shared<sg::camera>();

    // fov
    float fovx = 0.0f;
    if (doc.HasMember("fov"))
    {
        rapidjson::Value const& fov = doc["fov"];
        fovx = fov.GetFloat();
    }

    // aspect ratio
    float aspect = 0.0f;
    if (doc.HasMember("ratio"))
    {
        rapidjson::Value const& ratio = doc["fov"];
        aspect = ratio.GetFloat();
    }

    // name
    if (doc.HasMember("name"))
    {
        rapidjson::Value const& name = doc["name"];
        cam->name() = name.GetString();
    }

    // eye
    vec3 eye;
    if (doc.HasMember("eye"))
    {
        int i = 0;
        rapidjson::Value const& vec = doc["eye"];
        if (vec.IsArray())
        {
            for (auto& item : vec.GetArray())
            {
                eye[i++] = item.GetFloat();
                assert(i <= 3);
            }
        }
    }

    // focal length
    float focal_length = 0.0f;
    if (doc.HasMember("focalLength"))
    {
        rapidjson::Value const& fl = doc["focalLength"];
        focal_length = fl.GetFloat();
    }

    // center of interest / distance
    float distance = 0.0f;
    if (doc.HasMember("centerOfInterest"))
    {
        rapidjson::Value const& coi = doc["centerOfInterest"];
        distance = coi.GetFloat();
    }

    // lens radius
    float lens_radius = 0.0f;
    if (doc.HasMember("lensRadius"))
    {
        rapidjson::Value const& lr = doc["lensRadius"];
        lens_radius = lr.GetFloat();
    }

    // up
    vec3 up;
    if (doc.HasMember("up"))
    {
        int i = 0;
        rapidjson::Value const& vec = doc["up"];
        if (vec.IsArray())
        {
            for (auto& item : vec.GetArray())
            {
                up[i++] = item.GetFloat();
                assert(i <= 3);
            }
        }
    }

    // screenwindow
    if (doc.HasMember("screenwindow"))
    {
        float sw[4];

        int i = 0;
        rapidjson::Value const& vec = doc["screenwindow"];
        if (vec.IsArray())
        {
            for (auto& item : vec.GetArray())
            {
                sw[i++] = item.GetFloat();
                assert(i <= 4);
            }
        }
    }

    // look
    vec3 center;
    if (doc.HasMember("look"))
    {
        int i = 0;
        rapidjson::Value const& vec = doc["look"];
        if (vec.IsArray())
        {
            for (auto& item : vec.GetArray())
            {
                center[i++] = item.GetFloat();
                assert(i <= 3);
            }
        }
    }


    // Apply parameters
    float fovy = fovx / aspect;

    cam->perspective(fovy * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    cam->set_lens_radius(lens_radius);
    cam->set_focal_distance(focal_length);
    cam->look_at(eye, center, up);

    root->add_child(cam);
}


//-------------------------------------------------------------------------------------------------
// Load obj file
//

static void load_obj(
        boost::filesystem::path const& island_base_path,
        std::string const& filename,
        std::map<std::string, std::shared_ptr<sg::disney_material>> const& materials,
        std::map<std::string, std::shared_ptr<sg::texture>>& textures,
#if VSNRAY_COMMON_HAVE_PTEX
        std::shared_ptr<PtexPtr<PtexCache>> texture_cache,
#else
        void* texture_cache,
#endif
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

                face_id = 0;

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

                    auto tex = load_texture(texture_base_path, group + ".ptx", texture_cache);

                    if (tex != nullptr)
                    {
                        tex->name() = group;
                        textures.insert(std::make_pair(group, tex));
                        surf->add_texture(tex);
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
    }
}

static void load_instanced_primitive_json_file(
        boost::filesystem::path const& island_base_path,
        std::string const& filename,
        std::shared_ptr<sg::node> root,
        std::map<std::string, std::shared_ptr<sg::disney_material>>& materials,
        std::map<std::string, std::shared_ptr<sg::texture>>& textures,
#if VSNRAY_COMMON_HAVE_PTEX
        std::shared_ptr<PtexPtr<PtexCache>> texture_cache
#else
        void* texture_cache
#endif
        )
{
    std::cout << "Load instanced primitive json file: " << (island_base_path / filename).string() << '\n';
    cfile file((island_base_path / filename).string(), "r");
    if (!file.good())
    {
        std::cerr << "Cannot open " << filename << '\n';
        return;
    }

    char buffer[65536];
    rapidjson::FileReadStream frs(file.get(), buffer, sizeof(buffer));
    rapidjson::Document doc;
    doc.ParseStream(frs);

    for (auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it)
    {
        auto entry = it->value.GetObject();

        // Instance geometry
        std::string obj_file = it->name.GetString();
        std::vector<std::shared_ptr<sg::node>> objs;
        load_obj(island_base_path, obj_file, materials, textures, texture_cache, objs);

        // Instance transforms
        auto entries = it->value.GetObject();

        // Make room for child nodes
        size_t num_entries = entries.MemberEnd() - entries.MemberBegin();

        size_t root_first_child = root->children().size();
        size_t root_child_index = 0;

        root->children().resize(root->children().size() + num_entries);

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

            transform->children().resize(objs.size());
            for (size_t j = 0; j < objs.size(); ++j)
            {
                transform->children()[j] = objs[j];
            }
            root->children()[root_first_child + root_child_index++] = transform;
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
    cfile file(fn, "r");
    if (!file.good())
    {
        std::cerr << "Cannot open " << fn << '\n';
        return;
    }

    char buffer[65536];
    rapidjson::FileReadStream frs(file.get(), buffer, sizeof(buffer));
    rapidjson::Document doc;
    doc.ParseStream(frs);

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
                // All Moana colors are sRGB
                mat->base_color[i++] = std::pow(item.GetFloat(), 2.2f);
                assert(i <= 4);
            }
        }

        if (entry.HasMember("specTrans"))
        {
            rapidjson::Value const& spec_trans = entry["specTrans"];
            mat->spec_trans = spec_trans.GetFloat();
        }

        if (entry.HasMember("ior"))
        {
            rapidjson::Value const& ior = entry["ior"];
            mat->ior = ior.GetFloat();
        }

        if (entry.HasMember("refractive"))
        {
            rapidjson::Value const& refractive = entry["refractive"];
            mat->refractive = refractive.GetFloat();
        }

        if (entry.HasMember("colorMap"))
        {
            std::string path = entry["colorMap"].GetString();

            if (!path.empty())
            {
            }
        }

        // Replace red or magenta with white, these colors are *replaced* with the texture color
        if (mat->base_color.xyz() == vec3(1.0f, 0.0f, 0.0f) || mat->base_color.xyz() == vec3(1.0f, 0.0f, 1.0f))
        {
            mat->base_color.xyz() = vec3(1.0f, 1.0f, 1.0f);
        }

        materials.insert({ material_name, mat });
    }
}


//-------------------------------------------------------------------------------------------------
// Reset triangle mesh flags to 0
//

struct reset_flags_visitor : sg::node_visitor
{
    using node_visitor::apply;

    void apply(sg::surface_properties& sp)
    {
        sp.flags() = 0;

        node_visitor::apply(sp);
    }

    void apply(sg::triangle_mesh& tm)
    {
        tm.flags() = 0;

        node_visitor::apply(tm);
    }
};


//-------------------------------------------------------------------------------------------------
// Gather statistics
//

struct statistics_visitor : sg::node_visitor
{
    using node_visitor::apply;

    void apply(sg::node& n)
    {
        child_pointer_bytes += n.children().size() * sizeof(std::shared_ptr<sg::node>);
        parent_pointer_bytes += n.parents().size() * sizeof(std::weak_ptr<sg::node>);

        // Don't count twice (number of pure nodes is insignifanct)
        //node_bytes_total += sizeof(sg::node);

        node_visitor::apply(n);
    }

    void apply(sg::transform& tr)
    {
        matrix_bytes += sizeof(mat4);

        transform_node_bytes += sizeof(sg::transform);
        node_bytes_total += sizeof(sg::transform);

        apply(static_cast<sg::node&>(tr));
    }

    void apply(sg::surface_properties& sp)
    {
        if (sp.flags() == 0)
        {
            material_pointer_bytes += sizeof(std::shared_ptr<sg::material>);
            texture_pointer_bytes += sp.textures().size() * sizeof(std::shared_ptr<sg::texture>);

            if (std::dynamic_pointer_cast<sg::disney_material>(sp.material()))
            {
                material_bytes += sizeof(sg::disney_material);
            }
            else
            {
                material_bytes += sizeof(sg::material);
            }

            for (auto& t : sp.textures())
            {
#if VSNRAY_COMMON_HAVE_PTEX

                auto tex = std::dynamic_pointer_cast<sg::ptex_texture>(t);

                Ptex::String error = "";
                PtexPtr<PtexTexture> ptex_tex(tex->cache()->get()->get(tex->filename().c_str(), error));

                if (ptex_tex != nullptr)
                {
                    for (int faceid = 0; faceid < ptex_tex->numFaces(); ++faceid)
                    {
                        texture_bytes += Ptex::DataSize(ptex_tex->dataType())
                                       * ptex_tex->numChannels()
                                       * ptex_tex->getFaceInfo(faceid).res.size();
                    }
                }
#endif
            }

            surf_node_bytes += sizeof(sg::surface_properties);
            node_bytes_total += sizeof(sg::surface_properties);

            sp.flags() = ~sp.flags();
        }

        apply(static_cast<sg::node&>(sp));
    }

    void apply(sg::triangle_mesh& tm)
    {
        if (tm.flags() == 0)
        {
            vertices_bytes += tm.vertices.size() * sizeof(vec3);
            normals_bytes += tm.normals.size() * sizeof(vec3);
            tex_coords_bytes += tm.tex_coords.size() * sizeof(vec2);
            // No colors
            face_ids_bytes += tm.face_ids.size() * sizeof(int);

            mesh_node_bytes += sizeof(sg::triangle_mesh);
            node_bytes_total += sizeof(sg::triangle_mesh);

            tm.flags() = ~tm.flags(); // Don't count twice
        }

        apply(static_cast<sg::node&>(tm));
    }

    size_t vertices_bytes = 0;
    size_t normals_bytes = 0;
    size_t tex_coords_bytes = 0;
    size_t face_ids_bytes = 0;
    size_t matrix_bytes = 0;
    size_t material_bytes = 0;
    size_t texture_bytes = 0;
    size_t material_pointer_bytes = 0;
    size_t texture_pointer_bytes = 0;
    size_t child_pointer_bytes = 0;
    size_t parent_pointer_bytes = 0;
    size_t transform_node_bytes = 0;
    size_t surf_node_bytes = 0;
    size_t mesh_node_bytes = 0;
    size_t node_bytes_total = 0;
};

void load_moana(std::string const& filename, model& mod)
{
    std::vector<std::string> filenames(1);

    filenames[0] = filename;

    load_moana(filenames, mod);
}

void load_moana(std::vector<std::string> const& filenames, model& mod)
{
#if VSNRAY_COMMON_HAVE_PTEX
    std::shared_ptr<PtexPtr<PtexCache>> texture_cache = std::make_shared<PtexPtr<PtexCache>>(
        Ptex::PtexCache::create(
            1024,
            1ULL << 32, // 4GB
            true,       // premultiply (no alpha channel anyway)
            nullptr
            )
        );
#endif

    auto root = std::make_shared<sg::node>();

    for (auto filename : filenames)
    {
        // Extract base path
        boost::filesystem::path island_base_path = get_base_path(filename);

        if (island_base_path.empty())
        {
            std::cerr << "Cannot extract Moana Island Scene base path from " << filename << '\n';
            return;
        }

        // Handle special files
        boost::filesystem::path p(filename);
        boost::filesystem::path pp = p.parent_path();
        if (pp.filename().string() == "cameras")
        {
            load_camera(filename, root);
            continue;
        }

        // Materials map
        std::map<std::string, std::shared_ptr<sg::disney_material>> materials;

        // Textures, init with one empty texture
        std::map<std::string, std::shared_ptr<sg::texture>> textures;

#if VSNRAY_COMMON_HAVE_PTEX
        auto dummy = std::make_shared<sg::ptex_texture>("null", texture_cache);
#else
        // No texture cache..
        void* texture_cache = nullptr;
        // Add a 2D dummy texture
        auto dummy = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>();
        dummy->resize(1, 1);
        vector<4, unorm<8>> white(1.0f);
        dummy->reset(&white);
#endif
        dummy->name() = "null";
        textures.insert(std::make_pair("null", dummy));


        std::cout << "Load moana file: " << filename << '\n';
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


        // matFile
        if (doc.HasMember("matFile"))
        {
            std::string mat_file = doc["matFile"].GetString();
            load_material_file(island_base_path, mat_file, materials);
        }

        auto base_transform = std::make_shared<sg::transform>();
        root->add_child(base_transform);

        // transformMatrix
        if (doc.HasMember("transformMatrix"))
        {
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
        }


        // geomObjFile
        if (doc.HasMember("geomObjFile"))
        {
            std::string geom_obj_file = doc["geomObjFile"].GetString();
            std::vector<std::shared_ptr<sg::node>> objs;
            load_obj(island_base_path, geom_obj_file, materials, textures, texture_cache, objs);
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
                            textures,
                            texture_cache
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
                    load_obj(island_base_path, geom_obj_file, materials, textures, texture_cache, objs);
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
                                    textures,
                                    texture_cache
                                    );
                        }
                    }
                }
            }
        }
    }

    if (mod.scene_graph == nullptr)
    {
        mod.scene_graph = std::make_shared<sg::node>();
    }

    mod.scene_graph->add_child(root);

#if 1
    statistics_visitor stats_visitor;
    mod.scene_graph->accept(stats_visitor);

    static const size_t MB = 1024 * 1024;

    std::cout << "Vertices            (MB): " << stats_visitor.vertices_bytes / MB << '\n';
    std::cout << "Normals             (MB): " << stats_visitor.normals_bytes / MB << '\n';
    std::cout << "Texture coordinates (MB): " << stats_visitor.tex_coords_bytes / MB << '\n';
    std::cout << "Face IDs            (MB): " << stats_visitor.face_ids_bytes / MB << '\n';
    std::cout << "Matrices            (MB): " << stats_visitor.matrix_bytes / MB << '\n';
    std::cout << "Materials           (MB): " << stats_visitor.material_bytes / MB << '\n';
    std::cout << "Textures            (MB): " << stats_visitor.texture_bytes / MB << '\n';
    std::cout << "Material pointers   (MB): " << stats_visitor.material_pointer_bytes / MB << '\n';
    std::cout << "Texture pointers    (MB): " << stats_visitor.texture_pointer_bytes / MB << '\n';
    std::cout << "Child pointers      (MB): " << stats_visitor.child_pointer_bytes / MB << '\n';
    std::cout << "Parent pointers     (MB): " << stats_visitor.parent_pointer_bytes / MB << '\n';
    std::cout << "Transform nodes     (MB): " << stats_visitor.transform_node_bytes / MB << '\n';
    std::cout << "Surface nodes       (MB): " << stats_visitor.surf_node_bytes / MB << '\n';
    std::cout << "Mesh nodes          (MB): " << stats_visitor.mesh_node_bytes / MB << '\n';
    std::cout << "Nodes total         (MB): " << stats_visitor.node_bytes_total / MB << '\n';

    reset_flags_visitor reset_visitor;
    mod.scene_graph->accept(reset_visitor);
#endif
}

} // visionaray
