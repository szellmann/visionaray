// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <map>
#include <utility>

#include <boost/algorithm/string.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/utility/string_ref.hpp>
#include <boost/filesystem.hpp>

#include <visionaray/math/io.h>
#include <visionaray/math/vector.h>
#include <visionaray/texture/texture.h>

#include "image.h"
#include "make_texture.h"
#include "model.h"
#include "obj_grammar.h"
#include "obj_loader.h"

namespace qi = boost::spirit::qi;

using boost::string_ref;


namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Map obj indices to unsigned base-0 indices
//

template <typename Int>
inline Int remap_index(Int idx, Int size)
{
    return idx < 0 ? size + idx : idx - 1;
}


//-------------------------------------------------------------------------------------------------
// Store a triangle and assign visionaray-internal ids
//

static bool store_triangle(model& result, vertex_vector const& vertices, int i1, int i2, int i3)
{
    model::triangle_type tri;

    tri.v1 = vertices[i1];
    tri.e1 = vertices[i2] - tri.v1;
    tri.e2 = vertices[i3] - tri.v1;

    if (length(cross(tri.e1, tri.e2)) == 0.0f)
    {
        std::cerr << "Warning: rejecting degenerate triangle: zero-based indices: ("
                  << i1 << ' ' << i2 << ' ' << i3 << "), v1|e1|e2: "
                  << tri.v1 << ' ' << tri.e1 << ' ' << tri.e2 << '\n';
        return false;
    }
    else
    {
        tri.prim_id = static_cast<unsigned>(result.primitives.size());
        tri.geom_id = result.materials.size() == 0 ? 0 : static_cast<unsigned>(result.materials.size() - 1);
        result.primitives.push_back(tri);
    }

    return true;
}


//-------------------------------------------------------------------------------------------------
// Store obj faces (i.e. triangle fans) in vertex|tex_coords|normals lists
//

static void store_faces(
        model&                  result,
        vertex_vector const&    vertices,
        tex_coord_vector const& tex_coords,
        normal_vector const&    normals,
        face_vector const&      faces
        )
{

    auto vertices_size = static_cast<int>(vertices.size());
    size_t last = 2;
    auto i1 = remap_index(faces[0].vertex_index, vertices_size);

    while (last != faces.size())
    {
        // triangle
        auto i2 = remap_index(faces[last - 1].vertex_index, vertices_size);
        auto i3 = remap_index(faces[last].vertex_index, vertices_size);

        if (store_triangle(result, vertices, i1, i2, i3))
        {

            // texture coordinates
            if (faces[0].tex_coord_index && faces[last - 1].tex_coord_index && faces[last].tex_coord_index)
            {
                auto tex_coords_size = static_cast<int>(tex_coords.size());
                auto ti1 = remap_index(*faces[0].tex_coord_index, tex_coords_size);
                auto ti2 = remap_index(*faces[last - 1].tex_coord_index, tex_coords_size);
                auto ti3 = remap_index(*faces[last].tex_coord_index, tex_coords_size);

                result.tex_coords.push_back( tex_coords[ti1] );
                result.tex_coords.push_back( tex_coords[ti2] );
                result.tex_coords.push_back( tex_coords[ti3] );
            }

            // normals
            if (faces[0].normal_index && faces[last - 1].normal_index && faces[last].normal_index)
            {
                auto normals_size = static_cast<int>(normals.size());
                auto ni1 = remap_index(*faces[0].normal_index, normals_size);
                auto ni2 = remap_index(*faces[last - 1].normal_index, normals_size);
                auto ni3 = remap_index(*faces[last].normal_index, normals_size);

                result.shading_normals.push_back( normals[ni1] );
                result.shading_normals.push_back( normals[ni2] );
                result.shading_normals.push_back( normals[ni3] );
            }
        }

        ++last;
    }
}


//-------------------------------------------------------------------------------------------------
// aabb of a list of triangles
//

inline aabb bounds(model::triangle_list const& tris)
{
    aabb result;
    result.invalidate();

    for (auto const& tri : tris)
    {
        auto v1 = tri.v1;
        auto v2 = tri.v1 + tri.e1;
        auto v3 = tri.v1 + tri.e2;

        result = combine(result, v1);
        result = combine(result, v2);
        result = combine(result, v3);
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Obj material
//

struct mtl
{
    vec3 ka = vec3(0.2f, 0.2f, 0.2f);
    vec3 kd = vec3(0.8f, 0.8f, 0.8f);
    vec3 ke = vec3(0.0f, 0.0f, 0.0f);
    vec3 ks = vec3(0.1f, 0.1f, 0.1);
    float tr = 0.0f; // tr=1-d
    float d = 1.0f;
    float ns = 32.0f;
    float ni = 1.0f;
    std::string map_kd = "";
    int illum = 2;
};


//-------------------------------------------------------------------------------------------------
// Parse mtllib
//

static void parse_mtl(std::string const& filename, std::map<std::string, mtl>& matlib, obj_grammar const& grammar)
{
    boost::iostreams::mapped_file_source file(filename);

    std::map<std::string, mtl>::iterator mtl_it = matlib.end();

    string_ref text(file.data(), file.size());
    auto it = text.cbegin();

    string_ref mtl_name;

    while (it != text.cend())
    {
        if ( qi::phrase_parse(it, text.cend(), grammar.r_newmtl, qi::blank, mtl_name) )
        {
            std::string name(mtl_name.begin(), mtl_name.length());
            boost::trim(name);
            auto r = matlib.insert({ name, mtl() });
            if (!r.second)
            {
                // Material already exists...
            }

            mtl_it = r.first;
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), grammar.r_ka, qi::blank, mtl_it->second.ka) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), grammar.r_kd, qi::blank, mtl_it->second.kd) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), grammar.r_ke, qi::blank, mtl_it->second.ke) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), grammar.r_ks, qi::blank, mtl_it->second.ks) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), grammar.r_tr, qi::blank, mtl_it->second.tr) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), grammar.r_d, qi::blank, mtl_it->second.d) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), grammar.r_ns, qi::blank, mtl_it->second.ns) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), grammar.r_ni, qi::blank, mtl_it->second.ni) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), grammar.r_map_kd, qi::blank, mtl_it->second.map_kd) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), grammar.r_illum, qi::blank, mtl_it->second.illum) )
        {
        }
        else if ( qi::phrase_parse(it, text.cend(), grammar.r_unhandled, qi::blank) )
        {
        }
    }
}


//-------------------------------------------------------------------------------------------------
// Add material to container
//

template <typename Container>
void add_material(Container& cont, mtl m, string_ref name)
{
    model::material_type mat;
    mat.name() = std::string(name.data(), name.length());
    mat.ca = m.ka;
    mat.cd = m.kd;
    mat.cs = m.ks;
    mat.ce = m.ke;
    mat.ior = vec3(m.ni);
    mat.transmission = m.tr > 0.0f ? m.tr : 0.0f;
    mat.transmission = 1.0f - m.d > 0.0f ? 1.0f - m.d : mat.transmission;
    mat.specular_exp = m.ns;
    mat.illum = m.illum;
    cont.emplace_back(mat);
}


static void insert_dummy_texture(model& mod)
{
    using tex_type = model::texture_type;

    tex_type tex(1, 1);
    tex.set_address_mode(Wrap);
    tex.set_filter_mode(Nearest);

    vector<4, unorm<8>> dummy_texel(1.0f, 1.0f, 1.0f, 1.0f);
    tex.reset(&dummy_texel);

    mod.texture_map.insert(std::make_pair("null", std::move(tex)));

    // Maybe a "null" texture was already present and thus not inserted
    //  ==> find the one that was already inserted
    auto it = mod.texture_map.find("null");

    // Insert a ref
    mod.textures.push_back(tex_type::ref_type(it->second));
}


//-------------------------------------------------------------------------------------------------
// Load a single obj file
//

void load_obj(std::string const& filename, model& mod)
{
    std::vector<std::string> filenames(1);

    filenames[0] = filename;

    load_obj(filenames, mod);
}


//-------------------------------------------------------------------------------------------------
// Load obj files
//

void load_obj(std::vector<std::string> const& filenames, model& mod)
{
    std::vector<std::string> parsed_matlibs;

    std::map<std::string, mtl> matlib;

    size_t geom_id = 0;

    obj_grammar grammar;

    // containers for parsing

    string_ref comment;
    string_ref mtl_file;
    string_ref mtl_name;

    for (auto filename : filenames)
    {
        boost::iostreams::mapped_file_source file(filename);

        string_ref text(file.data(), file.size());
        auto it = text.cbegin();

        vertex_vector    vertices;
        tex_coord_vector tex_coords;
        normal_vector    normals;
        face_vector      faces;

        while (it != text.cend())
        {
            faces.clear();

            if ( qi::phrase_parse(it, text.cend(), grammar.r_comment, qi::blank, comment) )
            {
            }
            else if ( qi::phrase_parse(it, text.cend(), grammar.r_mtllib, qi::blank, mtl_file) )
            {
                std::string mtl_file_string(mtl_file.begin(), mtl_file.length());

                // Some obj files repeat the same mtllib command over and over again..
                bool already_parsed = std::find(parsed_matlibs.begin(), parsed_matlibs.end(), mtl_file_string) != parsed_matlibs.end();

                if (!already_parsed)
                {
                    boost::filesystem::path p(filename);
                    std::string mtl_dir = p.parent_path().string();

                    std::string mtl_path = "";
                    if (mtl_dir.empty())
                    {
                        mtl_path = std::string(mtl_file.begin(), mtl_file.length());
                    }
                    else
                    {
                        mtl_path = mtl_dir + "/" + std::string(mtl_file.begin(), mtl_file.length());
                    }

                    if (boost::filesystem::exists(mtl_path))
                    {
                        parse_mtl(mtl_path, matlib, grammar);
                    }
                    else
                    {
                        std::cerr << "Warning: file does not exist: " << mtl_path << '\n';
                    }

                    parsed_matlibs.push_back(mtl_file_string);
                }
                else
                {
                    std::cerr << "Warning: mtllib already parsed: " << mtl_file << '\n';
                }
            }
            else if ( qi::phrase_parse(it, text.cend(), grammar.r_usemtl, qi::blank, mtl_name) )
            {
                std::string name(mtl_name.begin(), mtl_name.length());
                boost::trim(name);
                auto mat_it = matlib.find(name);
                if (mat_it != matlib.end())
                {
                    typedef model::texture_type tex_type;

                    add_material(mod.materials, mat_it->second, name);

                    if (!mat_it->second.map_kd.empty()) // File path specified in mtl file
                    {
                        std::string tex_filename;

                        boost::filesystem::path kdp(mat_it->second.map_kd);

                        if (kdp.is_absolute())
                        {
                            tex_filename = kdp.string();
                        }

                        // Maybe boost::filesystem was wrong and a relative path
                        // camouflaged as an absolute one (e.g. because it was
                        // erroneously prefixed with a '/' under Unix.
                        // Happens e.g. in the fairy forest model..
                        // Let's also check for that..

                        if (!boost::filesystem::exists(tex_filename) || !kdp.is_absolute())
                        {
                            // Find texture relative to the path the obj file is located in
                            boost::filesystem::path p(filename);
                            tex_filename = p.parent_path().string() + "/" + mat_it->second.map_kd;
                            std::replace(tex_filename.begin(), tex_filename.end(), '\\', '/');
                        }

                        if (!boost::filesystem::exists(tex_filename))
                        {
                            boost::trim(tex_filename);
                        }

                        if (boost::filesystem::exists(tex_filename))
                        {
                            // Load the texture if we haven't done so yet
                            auto tex_it = mod.texture_map.find(mat_it->second.map_kd);
                            if (tex_it == mod.texture_map.end())
                            {
                                image img;
                                if (img.load(tex_filename))
                                {
                                    model::texture_type tex(img.width(), img.height());
                                    make_texture(tex, img);

                                    mod.texture_map.insert(std::make_pair(mat_it->second.map_kd, std::move(tex)));
                                    // Will be ref()'d below
                                    tex_it = mod.texture_map.find(mat_it->second.map_kd);
                                }
                                else
                                {
                                    std::cerr << "Warning: cannot load texture from file: " << tex_filename << '\n';
                                }
                            }

                            if (tex_it != mod.texture_map.end())
                            {
                                // File was already present in map or was
                                // just loaded. Push a reference to it!
                                auto& loaded_tex = tex_it->second;
                                mod.textures.push_back(tex_type::ref_type(loaded_tex));
                            }
                        }
                        else
                        {
                            std::cerr << "Warning: file does not exist: " << tex_filename << '\n';
                        }
                    }

                    // if no texture was loaded, insert a dummy
                    if (mod.textures.size() < mod.materials.size())
                    {
                        insert_dummy_texture(mod);
                    }

                    assert( mod.textures.size() == mod.materials.size() );
                }
                else
                {
                    std::cerr << "Warning: material not present in mtllib: " << name << '\n';
                }

                geom_id = mod.materials.size() == 0 ? 0 : mod.materials.size() - 1;
            }
            else if ( qi::phrase_parse(it, text.cend(), grammar.r_vertices, qi::blank, vertices) )
            {
            }
            else if ( qi::phrase_parse(it, text.cend(), grammar.r_tex_coords, qi::blank, tex_coords) )
            {
            }
            else if ( qi::phrase_parse(it, text.cend(), grammar.r_normals, qi::blank, normals) )
            {
            }
            else if ( qi::phrase_parse(it, text.cend(), grammar.r_face, qi::blank, faces) )
            {
                store_faces(mod, vertices, tex_coords, normals, faces);
            }
            else if ( qi::phrase_parse(it, text.cend(), grammar.r_unhandled, qi::blank) )
            {
            }
            else
            {
                ++it;
            }
        }

        // See that there is a material for each geometry
        for (size_t i = mod.materials.size(); i <= geom_id; ++i)
        {
            mod.materials.emplace_back(model::material_type());
        }

        // See that there is a (at least dummy) texture for each geometry
        for (size_t i = mod.textures.size(); i <= geom_id; ++i)
        {
            insert_dummy_texture(mod);
        }
    }

    // Calculate geometric normals
    for (auto const& tri : mod.primitives)
    {
        vec3 n = normalize(cross(tri.e1, tri.e2));
        mod.geometric_normals.push_back(n);
    }

    // See that each triangle has (potentially dummy) texture coordinates
    for (size_t i = mod.tex_coords.size(); i < mod.primitives.size() * 3; ++i)
    {
        mod.tex_coords.emplace_back(0.0f);
    }

	mod.bbox.insert(bounds(mod.primitives));
}

} // visionaray
