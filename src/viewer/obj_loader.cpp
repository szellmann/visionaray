// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <utility>

#include <boost/algorithm/string.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/utility/string_ref.hpp>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include <visionaray/detail/macros.h>
#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>

#include "image.h"
#include "obj_loader.h"

namespace qi = boost::spirit::qi;


//-------------------------------------------------------------------------------------------------
// boost::fusion-adapt some structs for parsing
//

namespace visionaray
{
namespace detail
{

struct face_index_t
{
    int vertex_index;
    boost::optional<int> tex_coord_index;
    boost::optional<int> normal_index;
};

} // detail
} // visionaray

BOOST_FUSION_ADAPT_STRUCT
(
    visionaray::detail::face_index_t,
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
}


namespace visionaray
{

typedef basic_triangle<3, float> triangle_type;

namespace detail
{


//-------------------------------------------------------------------------------------------------
// aabb of a list of triangles
//

aabb bounds(detail::triangle_list const& tris)
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


} // detail

struct mtl
{
    vec3 ka;
    vec3 kd;
    vec3 ks;
    float ns;
    std::string map_kd;
};


void parse_mtl(std::string const& filename, std::map<std::string, mtl>* matlib)
{
    std::ifstream ifstr(filename.c_str(), std::ifstream::in);
    std::string line;
    std::map<std::string, mtl>::iterator it;

    while (ifstr.good() && !ifstr.eof() && std::getline(ifstr, line))
    {
        boost::algorithm::trim(line);
        std::string identifier;
        std::istringstream str(line);
        str >> identifier >> std::ws;

        if (identifier == "newmtl")
        {
            std::string name;
            str >> name;
            auto mtl_pair = std::make_pair(name, mtl());
            matlib->insert(mtl_pair);
            it = matlib->find(name);
        }
        else if (identifier == "Ka")
        {
            str >> (*it).second.ka.x >> std::ws >> (*it).second.ka.y >> std::ws >> (*it).second.ka.z >> std::ws;
        }
        else if (identifier == "Kd")
        {
            str >> (*it).second.kd.x >> std::ws >> (*it).second.kd.y >> std::ws >> (*it).second.kd.z >> std::ws;
        }
        else if (identifier == "Ks")
        {
            str >> (*it).second.ks.x >> std::ws >> (*it).second.ks.y >> std::ws >> (*it).second.ks.z >> std::ws;
        }
        else if (identifier == "Ns")
        {
            str >> (*it).second.ns >> std::ws;
        }
        else if (identifier == "map_Kd")
        {
            str >> (*it).second.map_kd;
        }
    }
}


detail::obj_scene load_obj(std::string const& filename)
{
    std::map<std::string, mtl> matlib;

    std::ifstream ifstr(filename.c_str(), std::ifstream::in);
    boost::iostreams::mapped_file_source file(filename);

    unsigned geom_id = 0;

    using namespace detail;

    obj_scene result;

    using vertex_vector     = aligned_vector<vec3>;
    using tex_coord_vector  = aligned_vector<vec2>;
    using normal_vector     = aligned_vector<vec3>;
    using face_vector       = aligned_vector<face_index_t>;

    vertex_vector       vertices;
    tex_coord_vector    tex_coords;
    normal_vector       normals;
    face_vector         faces;


    // intermediate containers

    string_ref comment;
    string_ref mtl_file;
    string_ref mtl_name;


    // helper functions

    auto remap_index = [](int idx, int size) -> int
    {
        return idx < 0 ? static_cast<int>(size) + idx : idx - 1;
    };

    auto store_triangle = [&](int i1, int i2, int i3)
    {
        triangle_type tri;
        tri.prim_id = static_cast<unsigned>(result.primitives.size());
        tri.geom_id = geom_id;
        tri.v1 = vertices[i1];
        tri.e1 = vertices[i2] - tri.v1;
        tri.e2 = vertices[i3] - tri.v1;
        result.primitives.push_back(tri);
    };

    auto store_faces = [&](int vertices_size, int tex_coords_size, int normals_size)
    {
        size_t last = 2;
        auto i1 = remap_index(faces[0].vertex_index, vertices_size);

        while (last != faces.size())
        {
            // triangle
            auto i2 = remap_index(faces[last - 1].vertex_index, vertices_size);
            auto i3 = remap_index(faces[last].vertex_index, vertices_size);
            store_triangle(i1, i2, i3);

            // texture coordinates
            if (faces[0].tex_coord_index && faces[last - 1].tex_coord_index && faces[last].tex_coord_index)
            {
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
                auto ni1 = remap_index(*faces[0].normal_index, normals_size);
                auto ni2 = remap_index(*faces[last - 1].normal_index, normals_size);
                auto ni3 = remap_index(*faces[last].normal_index, normals_size);

                result.normals.push_back( normals[ni1] );
                result.normals.push_back( normals[ni2] );
                result.normals.push_back( normals[ni3] );
            }

            ++last;
        }
    };


    // obj grammar

    using It = string_ref::const_iterator;
    using space_type = decltype(qi::blank);

    qi::rule<It> r_unhandled                                    = *(qi::char_ - qi::eol)                                                    >> qi::eol;
    qi::rule<It, string_ref()> r_text_to_eol                    = qi::raw[*(qi::char_ - qi::eol)];

    qi::rule<It, string_ref()> r_comment                        = "#" >> r_text_to_eol                                                      >> qi::eol;
    qi::rule<It, string_ref(), space_type> r_mtllib             = "mtllib" >> r_text_to_eol                                                 >> qi::eol;
    qi::rule<It, string_ref(), space_type> r_usemtl             = "usemtl" >> r_text_to_eol                                                 >> qi::eol;

    qi::rule<It, vec3(), space_type> r_v                        = "v" >> qi::float_ >> qi::float_ >> qi::float_ >> -qi::float_              >> qi::eol; // TODO: mind w
    qi::rule<It, vec2(), space_type> r_vt                       = "vt" >> qi::float_ >> qi::float_ >> -qi::float_                           >> qi::eol; // TODO: mind w
    qi::rule<It, vec3(), space_type> r_vn                       = "vn" >> qi::float_ >> qi::float_ >> qi::float_                            >> qi::eol;

    qi::rule<It, vertex_vector(), space_type> r_vertices        = r_v >> *r_v;
    qi::rule<It, tex_coord_vector(), space_type> r_tex_coords   = r_vt >> *r_vt;
    qi::rule<It, normal_vector(), space_type> r_normals         = r_vn >> *r_vn;

    qi::rule<It, detail::face_index_t()> r_face_vertex          = qi::int_ >> -qi::lit("/") >> -qi::int_ >> -qi::lit("/") >> -qi::int_;
    qi::rule<It, face_vector(), space_type> r_face              = "f" >> r_face_vertex >> r_face_vertex >> r_face_vertex >> *r_face_vertex  >> qi::eol;


    string_ref text(file.data(), file.size());
    auto it = text.cbegin();

    while (it != text.cend())
    {
        faces.clear();

        if ( qi::phrase_parse(it, text.cend(), r_comment, qi::blank, comment) )
        {
            VSNRAY_UNUSED(comment);
            continue;
        }
        else if ( qi::phrase_parse(it, text.cend(), r_mtllib, qi::blank, mtl_file) )
        {
            boost::filesystem::path p(filename);
            std::string mtl_dir = p.parent_path().string();

            std::string mtl_path = mtl_dir + "/" + std::string(mtl_file);

            parse_mtl(mtl_path, &matlib);
        }
        else if ( qi::phrase_parse(it, text.cend(), r_usemtl, qi::blank, mtl_name) )
        {
            auto mat_it = matlib.find(std::string(mtl_name));
            if (mat_it != matlib.end())
            {
                phong<float> mat;
                mat.set_cd( mat_it->second.kd );
                mat.set_kd( 1.0f );
                mat.set_ks( 1.0f );
                mat.set_specular_exp( mat_it->second.ns );
                result.materials.push_back(mat);

                typedef tex_list::value_type tex_type;
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
                    tex.set_address_mode( Clamp );
                    tex.set_filter_mode( Linear );

                    auto data_ptr = reinterpret_cast<tex_type::value_type const*>(jpg.data());
                    tex.set_data(data_ptr);

                    result.textures.push_back(std::move(tex));
#endif
                }
                else
                {
                    result.textures.push_back(tex_type(0, 0));
                }
            }
            geom_id = result.materials.size() == 0 ? 0 : static_cast<unsigned>(result.materials.size() - 1);
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
            store_faces(static_cast<int>(vertices.size()), static_cast<int>(tex_coords.size()), static_cast<int>(normals.size()));
        }
        else if ( qi::phrase_parse(it, text.cend(), r_unhandled, qi::blank) )
        {
            continue;
        }
    }

// TODO

    result.normals.resize(0); // TODO: support for vertex normals
    if (result.normals.size() == 0) // have no default normals
    {
        for (auto const& tri : result.primitives)
        {
            vec3 n = normalize( cross(tri.e1, tri.e2) );
            result.normals.push_back(n);
        }
    }

    if (result.materials.size() == 0)
    {
        for (unsigned i = 0; i <= geom_id; ++i)
        {
            phong<float> m;
            m.set_cd( vec3(0.8f, 0.8f, 0.8f) );
            m.set_kd( 1.0f );
            m.set_ks( 1.0f );
            m.set_specular_exp( 32.0f );
            result.materials.push_back(m);
        }
    }
// TODO

    result.bbox = bounds(result.primitives);
    return result;
}

} // visionaray
