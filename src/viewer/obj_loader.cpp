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
#include <boost/filesystem.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/include/qi.hpp>

#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>

#include "image.h"
#include "obj_loader.h"

namespace qi = boost::spirit::qi;

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


std::pair<triangle_type, triangle_type> tesselate_quad(vec3 v1, vec3 v2, vec3 v3, vec3 v4)
{

    triangle_type t1;
    t1.v1 = v1;
    t1.e1 = v2 - t1.v1;
    t1.e2 = v3 - t1.v1;

    triangle_type t2;
    t2.v1 = v3;
    t2.e1 = v4 - t2.v1;
    t2.e2 = v1 - t2.v1;

    return std::make_pair(t1, t2);

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

    unsigned prim_id = 0;
    unsigned geom_id = 0;

    using namespace detail;
    obj_scene result;
    std::vector<vec3> vertices;
    std::vector<vec2> tex_coords;

    std::string line;
    while (ifstr.good() && !ifstr.eof() && std::getline(ifstr, line))
    {

        // comments

        if (line.length() > 0 && line[0] == '#')
        {
            continue;
        }

        // v, vn, vt, f, ...
        std::string identifier;
        std::istringstream str(line);
        str >> identifier >> std::ws;


        if (identifier.length() <= 0)
        {
            continue;
        }


        if (identifier == "mtllib")
        {
            boost::filesystem::path p(filename);
            std::string mtl_dir = p.parent_path().string();

            std::string mtl_file;
            str >> mtl_file >> std::ws;

            std::string mtl_path = mtl_dir + "/" + mtl_file;

            parse_mtl(mtl_path, &matlib);
        }


        if (identifier == "usemtl")
        {
            std::string mtl_name;
            str >> mtl_name >> std::ws;

            auto mat_it = matlib.find(mtl_name);
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


        vec3 v;
        vec3 n;
        vec2 t;

        if ( qi::phrase_parse(line.begin(), line.end(), "v" >> qi::float_ >> qi::float_ >> qi::float_, qi::space, v) )
        {
            vertices.push_back(v);
        }
        else if ( qi::phrase_parse(line.begin(), line.end(), "vn" >> qi::float_ >> qi::float_ >> qi::float_, qi::space, v) )
        {
            result.normals.push_back(n);
        }
        else if ( qi::phrase_parse(line.begin(), line.end(), "vt" >> qi::float_ >> qi::float_, qi::space, t) )
        {
            tex_coords.push_back(t);
        }
        else if (identifier == "f")
        {
            int cnt = 0;
            unsigned indices[4];
            unsigned tc_indices[4];

            bool has_tc = false;

            while (!str.eof())
            {
                int i;
                str >> i;

                // indices are either 1-based or negative
                if (i < 0)
                {
                    indices[cnt] = vertices.size() + i;
                }
                else
                {
                    indices[cnt] = i - 1;
                }

                if (str.get() == '/')
                {
                    if (str.peek() != '/')
                    {
                        has_tc = true;

                        int ti;
                        str >> ti;
                        tc_indices[cnt] = ti - 1;
                    }

                    if (str.get() == '/')
                    {
                        int ni;
                        str >> ni;
                        // TODO: handle normal indices
                    }
                }

                str >> std::ws;

                if (++cnt > 4)
                {
                    throw std::exception();
                    break;
                }
            }

            if (cnt == 4)
            {
                auto tris = detail::tesselate_quad(vertices[indices[0]], vertices[indices[1]], vertices[indices[2]], vertices[indices[3]]);

                tris.first.prim_id  = prim_id;
                tris.first.geom_id  = geom_id;
                ++prim_id;

                tris.second.prim_id = prim_id;
                tris.second.geom_id = geom_id;
                ++prim_id;

                result.primitives.push_back(tris.first);
                result.primitives.push_back(tris.second);

                if (has_tc)
                {
                    result.tex_coords.push_back(tex_coords[tc_indices[0]]);
                    result.tex_coords.push_back(tex_coords[tc_indices[1]]);
                    result.tex_coords.push_back(tex_coords[tc_indices[2]]);

                    result.tex_coords.push_back(tex_coords[tc_indices[2]]);
                    result.tex_coords.push_back(tex_coords[tc_indices[3]]);
                    result.tex_coords.push_back(tex_coords[tc_indices[0]]);
                }
            }
            else
            {
                triangle_type tri;
                tri.prim_id = prim_id;
                tri.geom_id = geom_id;
                tri.v1 = vertices[indices[0]];
                tri.e1 = vertices[indices[1]] - tri.v1;
                tri.e2 = vertices[indices[2]] - tri.v1;
                result.primitives.push_back(tri);
                ++prim_id;

                if (has_tc)
                {
                    result.tex_coords.push_back(tex_coords[tc_indices[0]]);
                    result.tex_coords.push_back(tex_coords[tc_indices[1]]);
                    result.tex_coords.push_back(tex_coords[tc_indices[2]]);
                }
            }
        }
    }

// TODO

    result.normals.resize(0);
//    if (result.normals.size() == 0) // have no default normals
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


