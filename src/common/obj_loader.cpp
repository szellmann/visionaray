// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <ostream>
#include <map>
#include <utility>

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/define_struct.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/utility/string_ref.hpp>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include <visionaray/math/vector.h>
#include <visionaray/texture/texture.h>

#include "image.h"
#include "model.h"
#include "obj_loader.h"

namespace qi = boost::spirit::qi;


//-------------------------------------------------------------------------------------------------
// boost::fusion-adapt/define some structs for parsing
//

BOOST_FUSION_DEFINE_STRUCT
(
    (visionaray), face_index_t,
    (int, vertex_index)
    (boost::optional<int>, tex_coord_index)
    (boost::optional<int>, normal_index)
)

BOOST_FUSION_ADAPT_STRUCT
(
    visionaray::vec2,
    (float, x)
    (float, y)
)

BOOST_FUSION_ADAPT_STRUCT
(
    visionaray::vec3,
    (float, x)
    (float, y)
    (float, z)
)


using boost::string_ref;

namespace boost
{
namespace spirit
{
namespace traits
{

template <typename Iterator, typename Enable>
struct assign_to_attribute_from_iterators<string_ref, Iterator, Enable>
{
    static void call(Iterator const& first, Iterator const& last, string_ref& attr)
    {
        attr = { first, static_cast<size_t>(last - first) };
    }
};

} // traits
} // spirit
} // boost


namespace visionaray
{

using triangle_type     = basic_triangle<3, float>;
using vertex_vector     = aligned_vector<vec3>;
using tex_coord_vector  = aligned_vector<vec2>;
using normal_vector     = aligned_vector<vec3>;
using face_vector       = aligned_vector<face_index_t>;


//-------------------------------------------------------------------------------------------------
// Default gray material
//

plastic<float> make_default_material()
{
    plastic<float> m;
    m.set_ca( from_rgb(0.2f, 0.2f, 0.2f) );
    m.set_cd( from_rgb(0.8f, 0.8f, 0.8f) );
    m.set_cs( from_rgb(0.1f, 0.1f, 0.1f) );
    m.set_ka( 1.0f );
    m.set_kd( 1.0f );
    m.set_ks( 1.0f );
    m.set_specular_exp( 32.0f );
    return m;
}


//-------------------------------------------------------------------------------------------------
// Map obj indices to unsigned base-0 indices
//

template <typename Int>
Int remap_index(Int idx, Int size)
{
    return idx < 0 ? size + idx : idx - 1;
}


//-------------------------------------------------------------------------------------------------
// Store a triangle and assign visionaray-internal ids
//

bool store_triangle(model& result, vertex_vector const& vertices, int i1, int i2, int i3)
{
    triangle_type tri;

    tri.v1 = vertices[i1];
    tri.e1 = vertices[i2] - tri.v1;
    tri.e2 = vertices[i3] - tri.v1;

    if (length(cross(tri.e1, tri.e2)) == 0.0f)
    {
        // TODO: implement some kind of error logging
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

void store_faces(
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

aabb bounds(model::triangle_list const& tris)
{
    aabb result( vec3(std::numeric_limits<float>::max()), -vec3(std::numeric_limits<float>::max()) );

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
    mtl() = default;
    mtl(plastic<float> m)
        : ka( to_rgb(m.get_ca() * m.get_ka()) )
        , kd( to_rgb(m.get_cd() * m.get_kd()) )
        , ke( vec3(0.0f, 0.0f, 0.0f) )
        , ks( to_rgb(m.get_cs() * m.get_ks()) )
        , ns(m.get_specular_exp())
    {
    }

    vec3 ka;
    vec3 kd;
    vec3 ke;
    vec3 ks;
    float ns;
    std::string map_kd;
};


//-------------------------------------------------------------------------------------------------
// Parse mtllib
//

void parse_mtl(std::string const& filename, std::map<std::string, mtl>& matlib)
{
    boost::iostreams::mapped_file_source file(filename);

    std::map<std::string, mtl>::iterator mtl_it = matlib.end();

    using It = string_ref::const_iterator;
    using skip_t = decltype(qi::blank);
    using sref_t = string_ref;
    using string = std::string;

    qi::rule<It> r_unhandled                = *(qi::char_ - qi::eol)                        >> qi::eol;
    qi::rule<It, sref_t(), skip_t> r_newmtl = "newmtl" >> qi::raw[*(qi::char_ - qi::eol)]   >> qi::eol;
    qi::rule<It, vec3(), skip_t> r_vec3     = qi::float_ >> qi::float_ >> qi::float_;
    qi::rule<It, vec3(), skip_t> r_ka       = "Ka" >> r_vec3                                >> qi::eol;
    qi::rule<It, vec3(), skip_t> r_kd       = "Kd" >> r_vec3                                >> qi::eol;
    qi::rule<It, vec3(), skip_t> r_ke       = "Ke" >> r_vec3                                >> qi::eol;
    qi::rule<It, vec3(), skip_t> r_ks       = "Ks" >> r_vec3                                >> qi::eol;
    qi::rule<It, float(), skip_t> r_ns      = "Ns" >> qi::float_                            >> qi::eol;
    qi::rule<It, string(), skip_t> r_map_kd = "map_Kd" >> *(qi::char_ - qi::eol)            >> qi::eol;

    string_ref text(file.data(), file.size());
    auto it = text.cbegin();

    string_ref mtl_name;

    while (it != text.cend())
    {
        if ( qi::phrase_parse(it, text.cend(), r_newmtl, qi::blank, mtl_name) )
        {
            auto r = matlib.insert({
                    std::string(mtl_name.begin(), mtl_name.length()),
                    mtl(make_default_material())}
                    );
            if (!r.second)
            {
                // Material already exists...
            }

            mtl_it = r.first;
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), r_ka, qi::blank, mtl_it->second.ka) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), r_kd, qi::blank, mtl_it->second.kd) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), r_ke, qi::blank, mtl_it->second.ke) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), r_ks, qi::blank, mtl_it->second.ks) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), r_ns, qi::blank, mtl_it->second.ns) )
        {
        }
        else if ( mtl_it != matlib.end() && qi::phrase_parse(it, text.cend(), r_map_kd, qi::blank, mtl_it->second.map_kd) )
        {
        }
        else if ( qi::phrase_parse(it, text.cend(), r_unhandled, qi::blank) )
        {
        }
    }
}

//-------------------------------------------------------------------------------------------------
// Add material to container
//

template <typename Container>
void add_material(
        plastic<float>  /* */,
        Container&      cont,
        mtl             m
        )
{
    plastic<float> mat;
    mat.set_ca( from_rgb(m.ka) );
    mat.set_cd( from_rgb(m.kd) );
    mat.set_cs( from_rgb(m.ks) );
    mat.set_ka( 1.0f );
    mat.set_kd( 1.0f );
    mat.set_ks( 1.0f );
    mat.set_specular_exp( m.ns );
    cont.emplace_back(mat);
}

template <typename ...Ts, typename Container>
void add_material(
        generic_material<Ts...> /* */,
        Container&              cont,
        mtl                     m
        )
{
    if (length(m.ke) > 0.0f)
    {
        // TODO: it is not guaranteed that generic_material
        // was intantiated with a parameter pack that
        // contains emissive<float> (but it is quite likely..)

        emissive<float> mat;
        mat.set_ce( from_rgb(m.ke) );
        mat.set_ls( 1.0f );
        cont.emplace_back(mat);
    }
    else
    {
        add_material(plastic<float>{}, cont, m);
    }
}


void load_obj(std::string const& filename, model& mod)
{
    std::map<std::string, mtl> matlib;

    boost::iostreams::mapped_file_source file(filename);

    size_t geom_id = 0;

    using It = string_ref::const_iterator;
    using skip_t = decltype(qi::blank);
    using sref_t = string_ref;
    using VV = vertex_vector;
    using TV = tex_coord_vector;
    using NV = normal_vector;
    using FV = face_vector;
    using FI = face_index_t;

    // obj grammar

    qi::rule<It> r_unhandled                = *(qi::char_ - qi::eol)                                                >> qi::eol;
    qi::rule<It, sref_t()> r_text_to_eol    = qi::raw[*(qi::char_ - qi::eol)];

    qi::rule<It, sref_t()> r_comment        = "#" >> r_text_to_eol                                                  >> qi::eol;
    qi::rule<It, sref_t(), skip_t> r_mtllib = "mtllib" >> r_text_to_eol                                             >> qi::eol;
    qi::rule<It, sref_t(), skip_t> r_usemtl = "usemtl" >> r_text_to_eol                                             >> qi::eol;

    qi::rule<It, vec3(), skip_t> r_v        = "v" >> qi::float_ >> qi::float_ >> qi::float_ >> -qi::float_          >> qi::eol  // TODO: mind w
                                            | "v" >> qi::float_ >> qi::float_ >> qi::float_
                                                      >> qi::float_ >> qi::float_ >> qi::float_                     >> qi::eol; // RGB color (extension)
    qi::rule<It, vec2(), skip_t> r_vt       = "vt" >> qi::float_ >> qi::float_ >> -qi::float_                       >> qi::eol; // TODO: mind w
    qi::rule<It, vec3(), skip_t> r_vn       = "vn" >> qi::float_ >> qi::float_ >> qi::float_                        >> qi::eol;

    qi::rule<It, VV(), skip_t> r_vertices   = r_v >> *r_v;
    qi::rule<It, TV(), skip_t> r_tex_coords = r_vt >> *r_vt;
    qi::rule<It, NV(), skip_t> r_normals    = r_vn >> *r_vn;

    qi::rule<It, FI()> r_face_idx           = qi::int_ >> -qi::lit("/") >> -qi::int_ >> -qi::lit("/") >> -qi::int_;
    qi::rule<It, FV(), skip_t> r_face       = "f" >> r_face_idx >> r_face_idx >> r_face_idx >> *r_face_idx          >> qi::eol;


    string_ref text(file.data(), file.size());
    auto it = text.cbegin();

    // containers for parsing

    vertex_vector       vertices;
    tex_coord_vector    tex_coords;
    normal_vector       normals;
    face_vector         faces;

    string_ref comment;
    string_ref mtl_file;
    string_ref mtl_name;


    while (it != text.cend())
    {
        faces.clear();

        if ( qi::phrase_parse(it, text.cend(), r_comment, qi::blank, comment) )
        {
        }
        else if ( qi::phrase_parse(it, text.cend(), r_mtllib, qi::blank, mtl_file) )
        {
            boost::filesystem::path p(filename);
            std::string mtl_dir = p.parent_path().string();

            std::string mtl_path = mtl_dir + "/" + std::string(mtl_file);

            if (boost::filesystem::exists(mtl_path))
            {
                parse_mtl(mtl_path, matlib);
            }
        }
        else if ( qi::phrase_parse(it, text.cend(), r_usemtl, qi::blank, mtl_name) )
        {
            auto mat_it = matlib.find(std::string(mtl_name));
            if (mat_it != matlib.end())
            {
                typedef model::texture_type tex_type;

                add_material(
                        model::material_type{},
                        mod.materials,
                        mat_it->second
                        );

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

                    if (boost::filesystem::exists(tex_filename))
                    {
                        // Load the texture if we haven't done so yet
                        auto tex_it = mod.texture_map.find(mat_it->second.map_kd);
                        if (tex_it == mod.texture_map.end())
                        {
                            image img;
                            if (img.load(tex_filename))
                            {
                                tex_type tex(img.width(), img.height());
                                tex.set_address_mode( Wrap );
                                tex.set_filter_mode( Linear );

                                if (img.format() == PF_RGB16UI)
                                {
                                    // Down-convert to 8-bit, add alpha=1.0
                                    auto data_ptr = reinterpret_cast<vector<3, unorm<16>> const*>(img.data());
                                    tex.reset(data_ptr, PF_RGB16UI, PF_RGBA8, AlphaIsOne);
                                }

                                else if (img.format() == PF_RGBA16UI)
                                {
                                    // Down-convert to 8-bit
                                    auto data_ptr = reinterpret_cast<vector<4, unorm<16>> const*>(img.data());
                                    tex.reset(data_ptr, PF_RGBA16UI, PF_RGBA8);
                                }
                                else if (img.format() == PF_R8)
                                {
                                    // Let RGB=R and add alpha=1.0
                                    auto data_ptr = reinterpret_cast<unorm< 8> const*>(img.data());
                                    tex.reset(data_ptr, PF_R8, PF_RGBA8, AlphaIsOne);
                                }
                                else if (img.format() == PF_RGB8)
                                {
                                    // Add alpha=1.0
                                    auto data_ptr = reinterpret_cast<vector<3, unorm< 8>> const*>(img.data());
                                    tex.reset(data_ptr, PF_RGB8, PF_RGBA8, AlphaIsOne);
                                }
                                else if (img.format() == PF_RGBA8)
                                {
                                    // "Native" texture format
                                    auto data_ptr = reinterpret_cast<vector<4, unorm< 8>> const*>(img.data());
                                    tex.reset(data_ptr);
                                }
                                else
                                {
                                    std::cerr << "Error: unsupported pixel format\n";
                                }

                                mod.texture_map.insert(std::make_pair(mat_it->second.map_kd, std::move(tex)));
                                // Will be ref()'d below
                                tex_it = mod.texture_map.find(mat_it->second.map_kd);
                            }
                            else
                            {
                                std::cerr << "Error: cannot load texture from file: " << tex_filename << '\n';
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
                        std::cerr << "Error: file does not exist: " << tex_filename << '\n';
                    }
                }

                // if no texture was loaded, insert an empty dummy
                if (mod.textures.size() < mod.materials.size())
                {
                    tex_type::ref_type tex(0, 0);
                    mod.textures.push_back(tex);
                }

                assert( mod.textures.size() == mod.materials.size() );
            }
            else
            {
                std::cerr << "Error: material not present in mtllib: " << mtl_name << '\n';
            }

            geom_id = mod.materials.size() == 0 ? 0 : mod.materials.size() - 1;
        }
        else if ( qi::phrase_parse(it, text.cend(), r_vertices, qi::blank, vertices) )
        {
        }
        else if ( qi::phrase_parse(it, text.cend(), r_tex_coords, qi::blank, tex_coords) )
        {
        }
        else if ( qi::phrase_parse(it, text.cend(), r_normals, qi::blank, normals) )
        {
        }
        else if ( qi::phrase_parse(it, text.cend(), r_face, qi::blank, faces) )
        {
            store_faces(mod, vertices, tex_coords, normals, faces);
        }
        else if ( qi::phrase_parse(it, text.cend(), r_unhandled, qi::blank) )
        {
        }
        else
        {
            ++it;
        }
    }


    // Calculate geometric normals
    for (auto const& tri : mod.primitives)
    {
        vec3 n = normalize( cross(tri.e1, tri.e2) );
        mod.geometric_normals.push_back(n);
    }

    // See that each triangle has (potentially dummy) texture coordinates
    for (size_t i = mod.tex_coords.size(); i < mod.primitives.size(); ++i)
    {
        mod.tex_coords.emplace_back(0.0f);
        mod.tex_coords.emplace_back(0.0f);
        mod.tex_coords.emplace_back(0.0f);
    }

    // See that there is a material for each geometry
    for (size_t i = mod.materials.size(); i <= geom_id; ++i)
    {
        mod.materials.emplace_back(make_default_material());
    }

    // See that there is a (at least dummy) texture for each geometry
    for (size_t i = mod.textures.size(); i <= geom_id; ++i)
    {
        using tex_type = model::texture_type;
        tex_type::ref_type tex(0, 0);
        mod.textures.push_back(tex);
    }

    mod.bbox = bounds(mod.primitives);
}

} // visionaray
