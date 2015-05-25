// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
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

void store_triangle(model& result, vertex_vector const& vertices, int i1, int i2, int i3)
{
    triangle_type tri;
    tri.prim_id = static_cast<unsigned>(result.primitives.size());
    tri.geom_id = result.materials.size() == 0 ? 0 : static_cast<unsigned>(result.materials.size() - 1);
    tri.v1 = vertices[i1];
    tri.e1 = vertices[i2] - tri.v1;
    tri.e2 = vertices[i3] - tri.v1;
    result.primitives.push_back(tri);
}


//-------------------------------------------------------------------------------------------------
// Store obj faces (i.e. triangle fans) in vertex|tex_coords|normals lists
//

void store_faces(model& result, vertex_vector const& vertices,
    tex_coord_vector const& tex_coords, normal_vector const& normals, face_vector const& faces)
{

    auto vertices_size = static_cast<int>(vertices.size());
    size_t last = 2;
    auto i1 = remap_index(faces[0].vertex_index, vertices_size);

    while (last != faces.size())
    {
        // triangle
        auto i2 = remap_index(faces[last - 1].vertex_index, vertices_size);
        auto i3 = remap_index(faces[last].vertex_index, vertices_size);
        store_triangle(result, vertices, i1, i2, i3);

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
/*        if (faces[0].normal_index && faces[last - 1].normal_index && faces[last].normal_index)
        {
            auto normals_size = static_cast<int>(normals.size());
            auto ni1 = remap_index(*faces[0].normal_index, normals_size);
            auto ni2 = remap_index(*faces[last - 1].normal_index, normals_size);
            auto ni3 = remap_index(*faces[last].normal_index, normals_size);

            result.normals.push_back( normals[ni1] );
            result.normals.push_back( normals[ni2] );
            result.normals.push_back( normals[ni3] );
        }*/

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

        result = combine(result, aabb(v1, v1));
        result = combine(result, aabb(v2, v2));
        result = combine(result, aabb(v3, v3));
    }

    return result;
}


struct mtl
{
    vec3 ka;
    vec3 kd;
    vec3 ks;
    float ns;
    std::string map_kd;
};


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
            auto r = matlib.insert({mtl_name.to_string(), mtl()});
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

            parse_mtl(mtl_path, matlib);
        }
        else if ( qi::phrase_parse(it, text.cend(), r_usemtl, qi::blank, mtl_name) )
        {
            auto mat_it = matlib.find(std::string(mtl_name));
            if (mat_it != matlib.end())
            {
                plastic<float> mat;
                mat.set_ca( from_rgb(mat_it->second.ka) );
                mat.set_cd( from_rgb(mat_it->second.kd) );
                mat.set_cs( from_rgb(mat_it->second.ks) );
                mat.set_ka( 1.0f );
                mat.set_kd( 1.0f );
                mat.set_ks( 1.0f );
                mat.set_specular_exp( mat_it->second.ns );
                mod.materials.push_back(mat);

                typedef model::tex_list::value_type tex_type;
                boost::filesystem::path p(filename);
                std::string tex_filename = p.parent_path().string() + "/" + mat_it->second.map_kd;

                static const std::string extensions[] = { ".jpg", ".jpeg", ".JPG", ".JPEG" };
                auto tex_path = boost::filesystem::path(tex_filename);
                auto has_jpg_ext = ( std::find(extensions, extensions + 4, tex_path.extension()) != extensions + 4 );

                if (!mat_it->second.map_kd.empty() && boost::filesystem::exists(tex_filename) && has_jpg_ext)
                {
#if defined(VSNRAY_HAVE_JPEG)
                    jpeg_image jpg(tex_filename);

                    tex_type tex(jpg.width(), jpg.height());
                    tex.set_address_mode( Wrap );
                    tex.set_filter_mode( Linear );

                    auto data_ptr = reinterpret_cast<tex_type::value_type const*>(jpg.data());
                    tex.set_data(data_ptr);

                    mod.textures.push_back(std::move(tex));
#endif
                }
                else
                {
                    mod.textures.push_back(tex_type(0, 0));
                }
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


// TODO
//  if (mod.normals.size() == 0) // have no default normals
    {
        for (auto const& tri : mod.primitives)
        {
            vec3 n = normalize( cross(tri.e1, tri.e2) );
            mod.normals.push_back(n);
        }
    }

    if (mod.materials.size() == 0)
    {
        for (size_t i = 0; i <= geom_id; ++i)
        {
            plastic<float> m;
            m.set_ca( from_rgb(0.2f, 0.2f, 0.2f) );
            m.set_cd( from_rgb(0.8f, 0.8f, 0.8f) );
            m.set_cs( from_rgb(0.1f, 0.1f, 0.1f) );
            m.set_ka( 1.0f );
            m.set_kd( 1.0f );
            m.set_ks( 1.0f );
            m.set_specular_exp( 32.0f );
            mod.materials.push_back(m);
        }
    }
// TODO

    mod.bbox = bounds(mod.primitives);
}

} // visionaray
