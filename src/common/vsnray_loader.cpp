// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/utility/string_ref.hpp>
#include <boost/assign.hpp>
#include <boost/bimap.hpp>
#include <boost/filesystem.hpp>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>

#include <visionaray/math/constants.h>
#include <visionaray/math/forward.h>
#include <visionaray/math/unorm.h>
#include <visionaray/math/vector.h>
#include <visionaray/texture/texture.h>

#include "cfile.h"
#include "image.h"
#include "model.h"
#include "sg.h"
#include "vsnray_loader.h"

using namespace visionaray;


namespace data_file
{

//-------------------------------------------------------------------------------------------------
// (included) data file meta data
//

struct meta_data
{
    enum encoding_t
    {
        Ascii,
        Binary
    };

    // VecN are binary compatible w/ visionaray::vecN
    enum data_type_t
    {
        U8,
        Float,
        Vec2u8,
        Vec2f,
        Vec3u8,
        Vec3f,
        Vec4u8,
        Vec4f,
    };

    static boost::bimap<data_type_t, std::string> data_type_map;

    enum compression_t
    {
        Raw
    };

    std::string   path;
    encoding_t    encoding    = Binary;
    data_type_t   data_type   = U8;
    int           num_items   = 0;
    compression_t compression = Raw;
    char          separator   = ' ';
};

boost::bimap<meta_data::data_type_t, std::string> meta_data::data_type_map
    = boost::assign::list_of<typename boost::bimap<meta_data::data_type_t, std::string>::relation>
        ( U8,     "u8" )
        ( Float,  "float" )
        ( Vec2u8, "vec2u8" )
        ( Vec2f,  "vec2f" )
        ( Vec3u8, "vec3u8" )
        ( Vec3f,  "vec3f" )
        ( Vec4u8, "vec4u8" )
        ( Vec4f,  "vec4f" );

} // data_file


//-------------------------------------------------------------------------------------------------
// Floating point number parser
//

template <typename It, typename Vector>
bool parse_floats(It first, It last, Vector& vec, char separator = ' ')
{
    namespace qi = boost::spirit::qi;

    return qi::phrase_parse(
            first,
            last,
            qi::float_ % *qi::char_(separator),
            qi::ascii::space,
            vec
            );
}

template <size_t N, typename Container>
bool parse_as_vecN(data_file::meta_data md, Container& vecNs)
{
    boost::iostreams::mapped_file_source file(md.path);

    if (md.data_type == data_file::meta_data::Float)
    {
        if (md.num_items % N != 0)
        {
            return false;
        }

        std::vector<float> floats;

        if (md.encoding == data_file::meta_data::Ascii)
        {
            boost::string_ref text(file.data(), file.size());

            parse_floats(text.cbegin(), text.cend(), floats, md.separator);

            if (static_cast<int>(floats.size()) != md.num_items)
            {
                return false;
            }
        }
        else // Binary
        {
            floats.resize(md.num_items);
            std::copy(
                file.data(),
                file.data() + file.size(),
                reinterpret_cast<char*>(floats.data())
                );
        }

        vecNs.resize(md.num_items / N);
        for (size_t i = 0; i < vecNs.size(); ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                vecNs[i][j] = floats[i * N + j];
            }
        }
    }
    else if (md.data_type == data_file::meta_data::Vec2u8)
    {
        if (N != 2)
        {
            throw std::runtime_error("");
        }

        if (md.encoding == data_file::meta_data::Ascii)
        {
            // Not implemented yet
            return false;
        }
        else // Binary
        {
            vecNs.resize(md.num_items);
            std::copy(
                file.data(),
                file.data() + file.size(),
                reinterpret_cast<char*>(vecNs.data())
                );
        }
    }
    else if (md.data_type == data_file::meta_data::Vec2f)
    {
        if (N != 2)
        {
            throw std::runtime_error("");
        }

        if (md.encoding == data_file::meta_data::Ascii)
        {
            // Not implemented yet
            return false;
        }
        else // Binary
        {
            vecNs.resize(md.num_items);
            std::copy(
                file.data(),
                file.data() + file.size(),
                reinterpret_cast<char*>(vecNs.data())
                );
        }
    }
    else if (md.data_type == data_file::meta_data::Vec3u8)
    {
        if (N != 3)
        {
            throw std::runtime_error("");
        }

        if (md.encoding == data_file::meta_data::Ascii)
        {
            // Not implemented yet
            return false;
        }
        else // Binary
        {
            vecNs.resize(md.num_items);
            std::copy(
                file.data(),
                file.data() + file.size(),
                reinterpret_cast<char*>(vecNs.data())
                );
        }
    }
    else if (md.data_type == data_file::meta_data::Vec3f)
    {
        if (N != 3)
        {
            throw std::runtime_error("");
        }

        if (md.encoding == data_file::meta_data::Ascii)
        {
            // Not implemented yet
            return false;
        }
        else // Binary
        {
            vecNs.resize(md.num_items);
            std::copy(
                file.data(),
                file.data() + file.size(),
                reinterpret_cast<char*>(vecNs.data())
                );
        }
    }
    else if (md.data_type == data_file::meta_data::Vec4u8)
    {
        if (N != 4)
        {
            throw std::runtime_error("");
        }

        if (md.encoding == data_file::meta_data::Ascii)
        {
            // Not implemented yet
            return false;
        }
        else // Binary
        {
            vecNs.resize(md.num_items);
            std::copy(
                file.data(),
                file.data() + file.size(),
                reinterpret_cast<char*>(vecNs.data())
                );
        }
    }
    else if (md.data_type == data_file::meta_data::Vec4f)
    {
        if (N != 4)
        {
            throw std::runtime_error("");
        }

        if (md.encoding == data_file::meta_data::Ascii)
        {
            // Not implemented yet
            return false;
        }
        else // Binary
        {
            vecNs.resize(md.num_items);
            std::copy(
                file.data(),
                file.data() + file.size(),
                reinterpret_cast<char*>(vecNs.data())
                );
        }
    }

    return true;
}

template <typename Container>
bool parse_as_vec2f(data_file::meta_data md, Container& vec2fs)
{
    return parse_as_vecN<2>(md, vec2fs);
}

template <typename Container>
bool parse_as_vec3f(data_file::meta_data md, Container& vec3fs)
{
    return parse_as_vecN<3>(md, vec3fs);
}


//-------------------------------------------------------------------------------------------------
// Parse tiny objects from json object
//

template <typename Object>
void parse_optional(float& f, Object const& obj, char const* member)
{
    if (obj.HasMember(member))
    {
        auto const& val = obj[member];

        if (!val.IsFloat())
        {
            throw std::runtime_error("");
        }

        f = val.GetFloat();
    }
}

template <size_t N, typename Object>
void parse_optional(vector<N, float>& v, Object const& obj, char const* member)
{
    if (obj.HasMember(member))
    {
        auto const& val = obj[member];

        if (!val.IsArray())
        {
            throw std::runtime_error("");
        }

        if (val.Capacity() != N)
        {
            throw std::runtime_error("");
        }

        for (size_t i = 0; i < N; ++i)
        {
            v[i] = val[i].GetFloat();
        }
    }
}

template <typename Object>
void parse_optional(recti& r, Object const& obj, char const* member)
{
    if (obj.HasMember(member))
    {
        auto const& val = obj[member];

        if (!val.IsArray())
        {
            throw std::runtime_error("");
        }

        if (val.Capacity() != 4)
        {
            throw std::runtime_error("");
        }

        for (size_t i = 0; i < 4; ++i)
        {
            r.data()[i] = val[i].GetFloat();
        }
    }
}


//-------------------------------------------------------------------------------------------------
// Parse texture parameters
//

template <typename Object>
bool parse_tex_address_mode_impl(Object const& obj, char const* attr, tex_address_mode& am)
{
    if (obj.HasMember(attr) && obj[attr].IsString())
    {
        std::string am_str = obj[attr].GetString();

        if (am_str == "wrap")
        {
            am = Wrap;
        }
        else if (am_str == "mirror")
        {
            am = Mirror;
        }
        else if (am_str == "clamp")
        {
            am = Clamp;
        }
        else if (am_str == "border")
        {
            am = Border;
        }
        else
        {
            throw std::runtime_error("");
        }

        return true;
    }

    return false;
}

// For 1-d textures
template <typename Object>
std::array<tex_address_mode, 1> parse_tex_address_mode1(Object const& obj)
{
    // Default: Wrap
    std::array<tex_address_mode, 1> result{{ Wrap }};

    tex_address_mode am = Wrap;

    // First parse address_mode
    if (parse_tex_address_mode_impl(obj, "address_mode", am))
    {
        result[0] = am;
    }

    // address_mode_s may override default / address_mode
    if (parse_tex_address_mode_impl(obj, "address_mode_s", am))
    {
        result[0] = am;
    }

    return result;
}

// For 2-d textures
template <typename Object>
std::array<tex_address_mode, 2> parse_tex_address_mode2(Object const& obj)
{
    // Default: Wrap
    std::array<tex_address_mode, 2> result{{ Wrap }};

    tex_address_mode am = Wrap;

    // First parse address_mode
    if (parse_tex_address_mode_impl(obj, "address_mode", am))
    {
        result[0] = am;
        result[1] = am;
    }

    // address_mode_{s|t} may override default / address_mode
    if (parse_tex_address_mode_impl(obj, "address_mode_s", am))
    {
        result[0] = am;
    }

    if (parse_tex_address_mode_impl(obj, "address_mode_t", am))
    {
        result[1] = am;
    }

    return result;
}

// For 3-d textures
template <typename Object>
std::array<tex_address_mode, 3> parse_tex_address_mode3(Object const& obj)
{
    // Default: Wrap
    std::array<tex_address_mode, 3> result{{ Wrap }};

    tex_address_mode am = Wrap;

    // First parse address_mode
    if (parse_tex_address_mode_impl(obj, "address_mode", am))
    {
        result[0] = am;
        result[1] = am;
        result[2] = am;
    }

    // address_mode_{s|t|r} may override default / address_mode
    if (parse_tex_address_mode_impl(obj, "address_mode_s", am))
    {
        result[0] = am;
    }

    if (parse_tex_address_mode_impl(obj, "address_mode_t", am))
    {
        result[1] = am;
    }

    if (parse_tex_address_mode_impl(obj, "address_mode_r", am))
    {
        result[2] = am;
    }

    return result;
}

template  <typename Object>
tex_filter_mode parse_tex_filter_mode(Object const& obj)
{
    if (obj.HasMember("filter_mode") && obj["filter_mode"].IsString())
    {
        std::string fm_str = obj["filter_mode"].GetString();

        if (fm_str == "nearest")
        {
            return Nearest;
        }
        else if (fm_str == "linear")
        {
            return Linear;
        }
        else if (fm_str == "bspline")
        {
            return BSpline;
        }
        else if (fm_str == "cardinalspline")
        {
            return CardinalSpline;
        }
        else
        {
            throw std::runtime_error("");
        }
    }

    return Linear;
}

template  <typename Object>
tex_color_space parse_tex_color_space(Object const& obj)
{
    if (obj.HasMember("color_space") && obj["color_space"].IsString())
    {
        std::string cs_str = obj["color_space"].GetString();

        if (cs_str == "rgb")
        {
            return RGB;
        }
        else if (cs_str == "srgb")
        {
            return sRGB;
        }
        else
        {
            throw std::runtime_error("");
        }
    }

    return RGB;
}


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
    std::shared_ptr<sg::node> parse_spot_light(Object const& obj);

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


    template <typename Object>
    data_file::meta_data parse_file_meta_data(Object const& obj);

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
        else if (strncmp(type_string.GetString(), "spot_light", 10) == 0)
        {
            result = parse_spot_light(obj);
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
    parse_optional(eye, obj, "eye");

    vec3 center(0.0f);
    parse_optional(center, obj, "center");

    vec3 up(0.0f);
    parse_optional(up, obj, "up");

    float fovy = 45.0f;
    parse_optional(fovy, obj, "fovy");

    float znear = 0.001f;
    parse_optional(znear, obj, "znear");

    float zfar = 1000.0f;
    parse_optional(zfar, obj, "zfar");

    recti viewport(0, 0, 0, 0);
    parse_optional(viewport, obj, "viewport");

    float lens_radius = 0.1f;
    parse_optional(lens_radius, obj, "lens_radius");

    float focal_distance = 10.0f;
    parse_optional(focal_distance, obj, "focal_distance");

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
    parse_optional(cl, obj, "cl");

    float kl = 1.0f;
    parse_optional(kl, obj, "kl");

    vec3 position(0.0f);
    parse_optional(position, obj, "position");

    float constant_attenuation = 1.0f;
    parse_optional(constant_attenuation, obj, "constant_attenuation");

    float linear_attenuation = 0.0f;
    parse_optional(linear_attenuation, obj, "linear_attenuation");

    float quadratic_attenuation = 0.0f;
    parse_optional(quadratic_attenuation, obj, "quadratic_attenuation");

    light->set_cl(cl);
    light->set_kl(kl);
    light->set_position(position);
    light->set_constant_attenuation(constant_attenuation);
    light->set_linear_attenuation(linear_attenuation);
    light->set_quadratic_attenuation(quadratic_attenuation);

    return light;
}

template <typename Object>
std::shared_ptr<sg::node> vsnray_parser::parse_spot_light(Object const& obj)
{
    auto light = std::make_shared<sg::spot_light>();

    vec3 cl(1.0f);
    parse_optional(cl, obj, "cl");

    float kl = 1.0f;
    parse_optional(kl, obj, "kl");

    vec3 position(0.0f);
    parse_optional(position, obj, "position");

    vec3 spot_direction(0.0f, 0.0f, -1.0f);
    parse_optional(spot_direction, obj, "spot_direction");
    assert(length(spot_direction) == 1.0f);

    float spot_cutoff = 180.0f * constants::degrees_to_radians<float>();
    parse_optional(spot_cutoff, obj, "spot_cutoff");

    float spot_exponent = 0.0f;
    parse_optional(spot_exponent, obj, "spot_exponent");

    float constant_attenuation = 1.0f;
    parse_optional(constant_attenuation, obj, "constant_attenuation");

    float linear_attenuation = 0.0f;
    parse_optional(linear_attenuation, obj, "linear_attenuation");

    float quadratic_attenuation = 0.0f;
    parse_optional(quadratic_attenuation, obj, "quadratic_attenuation");

    light->set_cl(cl);
    light->set_kl(kl);
    light->set_position(position);
    light->set_spot_direction(spot_direction);
    light->set_spot_cutoff(spot_cutoff);
    light->set_spot_exponent(spot_exponent);
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

        if (mat.Capacity() != 16)
        {
            throw std::runtime_error("");
        }

        for (rapidjson::SizeType i = 0; i < mat.Capacity(); ++i)
        {
            transform->matrix().data()[i] = mat[i].GetFloat();
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

                parse_optional(obj->ca, mat, "ca");
                parse_optional(obj->cd, mat, "cd");
                parse_optional(obj->cs, mat, "cs");
                parse_optional(obj->ce, mat, "ce");

                props->material() = obj;
            }
            else if (strncmp(type_string.GetString(), "glass", 5) == 0)
            {
                auto glass = std::make_shared<sg::glass_material>();

                parse_optional(glass->ct, mat, "ct");
                parse_optional(glass->cr, mat, "cr");
                parse_optional(glass->ior, mat, "ior");

                props->material() = glass;
            }
            else if (strncmp(type_string.GetString(), "disney", 6) == 0)
            {
                auto disney = std::make_shared<sg::disney_material>();

                parse_optional(disney->base_color, mat, "base_color");
                parse_optional(disney->spec_trans, mat, "spec_trans");
                parse_optional(disney->sheen, mat, "sheen");
                parse_optional(disney->sheen_tint, mat, "sheen_tint");
                parse_optional(disney->ior, mat, "ior");
                parse_optional(disney->refractive, mat, "refractive");
                parse_optional(disney->roughness, mat, "roughness");

                props->material() = disney;
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

    if (obj.HasMember("diffuse") && obj["diffuse"].IsObject())
    {
        std::string filename;

        auto diffuse_obj = obj["diffuse"].GetObject();

        bool can_load = diffuse_obj.HasMember("type")
                     && diffuse_obj["type"].IsString()
                     && diffuse_obj["type"].GetString() == std::string("texture2d")
                     && diffuse_obj.HasMember("filename")
                     && diffuse_obj["filename"].IsString();

        if (can_load)
        {
            filename = std::string(diffuse_obj["filename"].GetString());

            can_load &= boost::filesystem::exists(filename);
        }

        if (can_load)
        {
            image img;

            if (img.load(filename))
            {
                auto tex = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>();
                tex->resize(img.width(), img.height());
                tex->set_address_mode(parse_tex_address_mode2(diffuse_obj));
                tex->set_filter_mode(parse_tex_filter_mode(diffuse_obj));
                tex->set_color_space(parse_tex_color_space(diffuse_obj));

                // TODO: consolidate w/ obj loader
                if (img.format() == PF_RGB32F)
                {
                    // Down-convert to 8-bit, add alpha=1.0
                    auto data_ptr = reinterpret_cast<vector<3, float> const*>(img.data());
                    tex->reset(data_ptr, PF_RGB32F, PF_RGBA8, AlphaIsOne);
                }
                else if (img.format() == PF_RGBA32F)
                {
                    // Down-convert to 8-bit
                    auto data_ptr = reinterpret_cast<vector<4, float> const*>(img.data());
                    tex->reset(data_ptr, PF_RGBA32F, PF_RGBA8);
                }
                else if (img.format() == PF_RGB16UI)
                {
                    // Down-convert to 8-bit, add alpha=1.0
                    auto data_ptr = reinterpret_cast<vector<3, unorm<16>> const*>(img.data());
                    tex->reset(data_ptr, PF_RGB16UI, PF_RGBA8, AlphaIsOne);
                }
                else if (img.format() == PF_RGBA16UI)
                {
                    // Down-convert to 8-bit
                    auto data_ptr = reinterpret_cast<vector<4, unorm<16>> const*>(img.data());
                    tex->reset(data_ptr, PF_RGBA16UI, PF_RGBA8);
                }
                else if (img.format() == PF_R8)
                {
                    // Let RGB=R and add alpha=1.0
                    auto data_ptr = reinterpret_cast<unorm< 8> const*>(img.data());
                    tex->reset(data_ptr, PF_R8, PF_RGBA8, AlphaIsOne);
                }
                else if (img.format() == PF_RGB8)
                {
                    // Add alpha=1.0
                    auto data_ptr = reinterpret_cast<vector<3, unorm< 8>> const*>(img.data());
                    tex->reset(data_ptr, PF_RGB8, PF_RGBA8, AlphaIsOne);
                }
                else if (img.format() == PF_RGBA8)
                {
                    // "Native" texture format
                    auto data_ptr = reinterpret_cast<vector<4, unorm< 8>> const*>(img.data());
                    tex->reset(data_ptr);
                }
                else
                {
                    throw std::runtime_error("");
                }

                props->add_texture(tex);
            }
        }
    }

    if (props->textures().size() == 0)
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

        if (verts.IsArray())
        {
            for (rapidjson::SizeType i = 0; i < verts.Capacity(); i += 3)
            {
                mesh->vertices.emplace_back(
                    verts[i].GetFloat(),
                    verts[i + 1].GetFloat(),
                    verts[i + 2].GetFloat()
                    );
            }
        }
        else if (verts.IsObject())
        {
            auto const& type_string = verts["type"];
            if (strncmp(type_string.GetString(), "file", 4) == 0)
            {
                auto md = parse_file_meta_data(verts);

                if (!parse_as_vec3f(md, mesh->vertices))
                {
                    throw std::runtime_error("");
                }
            }
        }
        else
        {
            throw std::runtime_error("");
        }
    }

    if (obj.HasMember("normals"))
    {
        auto const& normals = obj["normals"];

        if (normals.IsArray())
        {
            for (rapidjson::SizeType i = 0; i < normals.Capacity(); i += 3)
            {
                mesh->normals.emplace_back(
                    normals[i].GetFloat(),
                    normals[i + 1].GetFloat(),
                    normals[i + 2].GetFloat()
                    );
            }
        }
        else if (normals.IsObject())
        {
            auto const& type_string = normals["type"];
            if (strncmp(type_string.GetString(), "file", 4) == 0)
            {
                auto md = parse_file_meta_data(normals);

                if (!parse_as_vec3f(md, mesh->normals))
                {
                    throw std::runtime_error("");
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
        for (size_t i = 0; i < mesh->vertices.size(); i += 3)
        {
            vec3 v1 = mesh->vertices[i];
            vec3 v2 = mesh->vertices[i + 1];
            vec3 v3 = mesh->vertices[i + 2];

            vec3 e1 = v2 - v1;
            vec3 e2 = v3 - v1;

            vec3 gn = normalize(cross(e1, e2));

            mesh->normals.emplace_back(gn);
            mesh->normals.emplace_back(gn);
            mesh->normals.emplace_back(gn);
        }
    }

    if (obj.HasMember("tex_coords"))
    {
        auto const& tex_coords = obj["tex_coords"];

        if (tex_coords.IsArray())
        {
            for (rapidjson::SizeType i = 0; i < tex_coords.Capacity(); i += 2)
            {
                mesh->tex_coords.emplace_back(
                    tex_coords[i].GetFloat(),
                    tex_coords[i + 1].GetFloat()
                    );
            }
        }
        else if (tex_coords.IsObject())
        {
            auto const& type_string = tex_coords["type"];
            if (strncmp(type_string.GetString(), "file", 4) == 0)
            {
                auto md = parse_file_meta_data(tex_coords);

                if (!parse_as_vec2f(md, mesh->tex_coords))
                {
                    throw std::runtime_error("");
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
        for (size_t i = 0; i < mesh->vertices.size(); i += 3)
        {
            mesh->tex_coords.emplace_back(0.0f, 0.0f);
            mesh->tex_coords.emplace_back(0.0f, 0.0f);
            mesh->tex_coords.emplace_back(0.0f, 0.0f);
        }
    }

    if (obj.HasMember("colors"))
    {
        auto const& colors = obj["colors"];

        if (colors.IsArray())
        {
            for (rapidjson::SizeType i = 0; i < colors.Capacity(); i += 3)
            {
                mesh->colors.emplace_back(
                    colors[i].GetFloat(),
                    colors[i + 1].GetFloat(),
                    colors[i + 2].GetFloat()
                    );
            }
        }
        else if (colors.IsObject())
        {
            auto const& type_string = colors["type"];
            if (strncmp(type_string.GetString(), "file", 4) == 0)
            {
                auto md = parse_file_meta_data(colors);

                if (!parse_as_vec3f(md, mesh->colors))
                {
                    throw std::runtime_error("");
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
        for (size_t i = 0; i < mesh->vertices.size(); i += 3)
        {
            mesh->colors.emplace_back(1.0f, 1.0f, 1.0f);
            mesh->colors.emplace_back(1.0f, 1.0f, 1.0f);
            mesh->colors.emplace_back(1.0f, 1.0f, 1.0f);
        }
    }

    return mesh;
}

template <typename Object>
std::shared_ptr<sg::node> vsnray_parser::parse_indexed_triangle_mesh(Object const& obj)
{
    auto mesh = std::make_shared<sg::indexed_triangle_mesh>();

    if (obj.HasMember("vertex_indices"))
    {
        auto const& vertex_indices = obj["vertex_indices"];

        for (auto const& item : vertex_indices.GetArray())
        {
            mesh->vertex_indices.push_back(item.GetInt());
        }
    }

    if (obj.HasMember("normal_indices"))
    {
        auto const& normal_indices = obj["normal_indices"];

        for (auto const& item : normal_indices.GetArray())
        {
            mesh->normal_indices.push_back(item.GetInt());
        }
    }

    if (obj.HasMember("tex_coord_indices"))
    {
        auto const& tex_coord_indices = obj["tex_coord_indices"];

        for (auto const& item : tex_coord_indices.GetArray())
        {
            mesh->tex_coord_indices.push_back(item.GetInt());
        }
    }

    if (obj.HasMember("color_indices"))
    {
        auto const& color_indices = obj["color_indices"];

        for (auto const& item : color_indices.GetArray())
        {
            mesh->color_indices.push_back(item.GetInt());
        }
    }

    if (obj.HasMember("vertices"))
    {
        mesh->vertices = std::make_shared<aligned_vector<vec3>>();

        auto const& verts = obj["vertices"];

        if (verts.IsArray())
        {
            for (rapidjson::SizeType i = 0; i < verts.Capacity(); i += 3)
            {
                mesh->vertices->emplace_back(
                    verts[i].GetFloat(),
                    verts[i + 1].GetFloat(),
                    verts[i + 2].GetFloat()
                    );
            }
        }
        else if (verts.IsObject())
        {
            auto const& type_string = verts["type"];
            if (strncmp(type_string.GetString(), "file", 4) == 0)
            {
                auto md = parse_file_meta_data(verts);

                if (!parse_as_vec3f(md, *mesh->vertices))
                {
                    throw std::runtime_error("");
                }
            }
        }
    }

    if (obj.HasMember("normals"))
    {
        mesh->normals = std::make_shared<aligned_vector<vec3>>();

        auto const& normals = obj["normals"];

        if (normals.IsArray())
        {
            for (rapidjson::SizeType i = 0; i < normals.Capacity(); i += 3)
            {
                mesh->normals->emplace_back(
                    normals[i].GetFloat(),
                    normals[i + 1].GetFloat(),
                    normals[i + 2].GetFloat()
                    );
            }
        }
        else if (normals.IsObject())
        {
            auto const& type_string = normals["type"];
            if (strncmp(type_string.GetString(), "file", 4) == 0)
            {
                auto md = parse_file_meta_data(normals);

                if (!parse_as_vec3f(md, *mesh->normals))
                {
                    throw std::runtime_error("");
                }
            }
        }
        else
        {
            throw std::runtime_error("");
        }
    }

    if (obj.HasMember("tex_coords"))
    {
        mesh->tex_coords = std::make_shared<aligned_vector<vec2>>();

        auto const& tex_coords = obj["tex_coords"];

        if (tex_coords.IsArray())
        {
            for (rapidjson::SizeType i = 0; i < tex_coords.Capacity(); i += 2)
            {
                mesh->tex_coords->emplace_back(
                    tex_coords[i].GetFloat(),
                    tex_coords[i + 1].GetFloat()
                    );
            }
        }
        else if (tex_coords.IsObject())
        {
            auto const& type_string = tex_coords["type"];
            if (strncmp(type_string.GetString(), "file", 4) == 0)
            {
                auto md = parse_file_meta_data(tex_coords);

                if (!parse_as_vec2f(md, *mesh->tex_coords))
                {
                    throw std::runtime_error("");
                }
            }
        }
        else
        {
            throw std::runtime_error("");
        }
    }

    if (obj.HasMember("colors"))
    {
        mesh->colors = std::make_shared<aligned_vector<vector<3, unorm<8>>>>();

        auto const& colors = obj["colors"];

        if (colors.IsArray())
        {
            for (rapidjson::SizeType i = 0; i < colors.Capacity(); i += 3)
            {
                mesh->colors->emplace_back(
                    colors[i].GetFloat(),
                    colors[i + 1].GetFloat(),
                    colors[i + 2].GetFloat()
                    );
            }
        }
        else if (colors.IsObject())
        {
            auto const& type_string = colors["type"];
            if (strncmp(type_string.GetString(), "file", 4) == 0)
            {
                auto md = parse_file_meta_data(colors);

                if (!parse_as_vec3f(md, *mesh->colors))
                {
                    throw std::runtime_error("");
                }
            }
        }
        else
        {
            throw std::runtime_error("");
        }
    }

    return mesh;
}

template <typename Object>
data_file::meta_data vsnray_parser::parse_file_meta_data(Object const& obj)
{
    data_file::meta_data result;

    if (obj.HasMember("path"))
    {
        result.path = obj["path"].GetString();
    }
    else
    {
        throw std::runtime_error("");
    }

    if (obj.HasMember("encoding"))
    {
        std::string encoding = obj["encoding"].GetString();
        if (encoding == "ascii")
        {
            result.encoding = data_file::meta_data::Ascii;
        }
        else if (encoding == "binary")
        {
            result.encoding = data_file::meta_data::Binary;
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

    if (obj.HasMember("data_type"))
    {
        std::string data_type = obj["data_type"].GetString();

        auto& data_type_map = data_file::meta_data::data_type_map;

        auto it = data_type_map.right.find(data_type);

        if (it != data_type_map.right.end())
        {
            result.data_type = it->second;
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

    if (obj.HasMember("num_items"))
    {
        result.num_items = obj["num_items"].GetInt();
    }
    else
    {
        throw std::runtime_error("");
    }

    if (obj.HasMember("compression"))
    {
        std::string compression = obj["compression"].GetString();
        if (compression == "none" || compression == "raw")
        {
            result.compression = data_file::meta_data::Raw;
        }
        else
        {
            throw std::runtime_error("");
        }
    }

    if (obj.HasMember("separator"))
    {
        std::string separator = obj["separator"].GetString();
        result.separator = separator[0];
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// .vsnray writer
//

class vsnray_writer
{
public:

    vsnray_writer(rapidjson::Document& doc, std::string filename)
        : document_(doc)
        , filename_(filename)
    {
    }

    template <typename Object>
    void write_node(Object obj, std::shared_ptr<sg::node> const& n);

    template <typename Object>
    void write_transform(Object obj, std::shared_ptr<sg::transform> const& tr);

    template <typename Object>
    void write_surface_properties(Object obj, std::shared_ptr<sg::surface_properties> const& sp);

    template <typename Object>
    void write_triangle_mesh(Object obj, std::shared_ptr<sg::triangle_mesh> const& tm);

    template <typename Object>
    void write_indexed_triangle_mesh(Object obj, std::shared_ptr<sg::indexed_triangle_mesh> const& tm);


    template <typename Object, typename Container>
    void write_data_file(Object obj, data_file::meta_data md, Container const& cont);

private:

    std::string make_inline_filename(std::string node_name, std::string suffix);

    rapidjson::Document& document_;

    std::string filename_;

};

//-------------------------------------------------------------------------------------------------
// Write nodes
//

template <typename Object>
void vsnray_writer::write_node(Object obj, std::shared_ptr<sg::node> const& n)
{
    if (n == nullptr)
    {
        return;
    }

    auto& allocator = document_.GetAllocator();

    rapidjson::Value name(n->name().c_str(), allocator);
    obj.AddMember(
        rapidjson::StringRef("name"),
        name,
        allocator
        );

    if (auto tr = std::dynamic_pointer_cast<sg::transform>(n))
    {
        obj.AddMember(
            rapidjson::StringRef("type"),
            rapidjson::StringRef("transform"),
            allocator
            );

        write_transform(obj, tr);
    }
    else if (auto sp = std::dynamic_pointer_cast<sg::surface_properties>(n))
    {
        obj.AddMember(
            rapidjson::StringRef("type"),
            rapidjson::StringRef("surface_properties"),
            allocator
            );

        write_surface_properties(obj, sp);
    }
    else if (auto tm = std::dynamic_pointer_cast<sg::triangle_mesh>(n))
    {
        if (tm->flags() == 0)
        {
            obj.AddMember(
                rapidjson::StringRef("type"),
                rapidjson::StringRef("triangle_mesh"),
                allocator
                );

            write_triangle_mesh(obj, tm);

            tm->flags() = 1;
        }
    }
    else if (auto itm = std::dynamic_pointer_cast<sg::indexed_triangle_mesh>(n))
    {
        obj.AddMember(
            rapidjson::StringRef("type"),
            rapidjson::StringRef("indexed_triangle_mesh"),
            allocator
            );

        write_indexed_triangle_mesh(obj, itm);
    }
    else
    {
        obj.AddMember(
            rapidjson::StringRef("type"),
            rapidjson::StringRef("node"),
            allocator
            );
    }

    if (n->children().size() > 0)
    {
        rapidjson::Value arr(rapidjson::kArrayType);

        for (auto& c : n->children())
        {
            rapidjson::Value child;
            child.SetObject();
            write_node(child.GetObject(), c);

            arr.PushBack(child, allocator);
        }

        obj.AddMember("children", arr, allocator);
    }
}

template <typename Object>
void vsnray_writer::write_transform(Object obj, std::shared_ptr<sg::transform> const& tr)
{
    auto& allocator = document_.GetAllocator();

    rapidjson::Value mat(rapidjson::kArrayType);

    for (int i = 0; i < 16; ++i)
    {
        rapidjson::Value val;
        val.SetDouble(tr->matrix().data()[i]);
        mat.PushBack(val, allocator);
    }

    obj.AddMember("matrix", mat, allocator);
}

template <typename Object>
void vsnray_writer::write_surface_properties(Object obj, std::shared_ptr<sg::surface_properties> const& sp)
{
    auto& allocator = document_.GetAllocator();

    if (sp->material() != nullptr)
    {
        rapidjson::Value name(sp->material()->name().c_str(), allocator);
        obj.AddMember(
            rapidjson::StringRef("name"),
            name,
            allocator
            );

        rapidjson::Value jmat;
        jmat.SetObject();

        if (auto mat = std::dynamic_pointer_cast<sg::obj_material>(sp->material()))
        {
            jmat.AddMember(
                rapidjson::StringRef("type"),
                rapidjson::StringRef("obj"),
                allocator
                );

            rapidjson::Value ca(rapidjson::kArrayType);
            rapidjson::Value cd(rapidjson::kArrayType);
            rapidjson::Value cs(rapidjson::kArrayType);
            rapidjson::Value ce(rapidjson::kArrayType);

            for (int i = 0; i < 3; ++i)
            {
                ca.PushBack(rapidjson::Value().SetFloat(mat->ca[i]), allocator);
                cd.PushBack(rapidjson::Value().SetFloat(mat->cd[i]), allocator);
                cs.PushBack(rapidjson::Value().SetFloat(mat->cs[i]), allocator);
                ce.PushBack(rapidjson::Value().SetFloat(mat->ce[i]), allocator);
            }

            jmat.AddMember(rapidjson::StringRef("ca"), ca, allocator);
            jmat.AddMember(rapidjson::StringRef("cd"), cd, allocator);
            jmat.AddMember(rapidjson::StringRef("cs"), cs, allocator);
            jmat.AddMember(rapidjson::StringRef("ce"), ce, allocator);
        }
        else if (auto mat = std::dynamic_pointer_cast<sg::glass_material>(sp->material()))
        {
            jmat.AddMember(
                rapidjson::StringRef("type"),
                rapidjson::StringRef("glass"),
                allocator
                );

            rapidjson::Value ct(rapidjson::kArrayType);
            rapidjson::Value cr(rapidjson::kArrayType);
            rapidjson::Value ior(rapidjson::kArrayType);

            for (int i = 0; i < 3; ++i)
            {
                ct.PushBack(rapidjson::Value().SetFloat(mat->ct[i]), allocator);
                cr.PushBack(rapidjson::Value().SetFloat(mat->cr[i]), allocator);
                ior.PushBack(rapidjson::Value().SetFloat(mat->ior[i]), allocator);
            }

            jmat.AddMember(rapidjson::StringRef("ct"), ct, allocator);
            jmat.AddMember(rapidjson::StringRef("cr"), cr, allocator);
            jmat.AddMember(rapidjson::StringRef("ior"), ior, allocator);
        }
        else if (auto mat = std::dynamic_pointer_cast<sg::disney_material>(sp->material()))
        {
            jmat.AddMember(
                rapidjson::StringRef("type"),
                rapidjson::StringRef("disney"),
                allocator
                );

            rapidjson::Value base_color(rapidjson::kArrayType);

            for (int i = 0; i < 4; ++i)
            {
                base_color.PushBack(rapidjson::Value().SetFloat(mat->base_color[i]), allocator);
            }

            jmat.AddMember(rapidjson::StringRef("base_color"), base_color, allocator);
            jmat.AddMember(
                rapidjson::StringRef("spec_trans"),
                rapidjson::Value().SetFloat(mat->spec_trans),
                allocator
                );
            jmat.AddMember(
                rapidjson::StringRef("sheen"),
                rapidjson::Value().SetFloat(mat->sheen),
                allocator
                );
            jmat.AddMember(
                rapidjson::StringRef("sheen_tint"),
                rapidjson::Value().SetFloat(mat->sheen_tint),
                allocator
                );
            jmat.AddMember(
                rapidjson::StringRef("ior"),
                rapidjson::Value().SetFloat(mat->ior),
                allocator
                );
            jmat.AddMember(
                rapidjson::StringRef("refractive"),
                rapidjson::Value().SetFloat(mat->refractive),
                allocator
                );
            jmat.AddMember(
                rapidjson::StringRef("roughness"),
                rapidjson::Value().SetFloat(mat->roughness),
                allocator
                );
        }
        else
        {
            throw std::runtime_error("");
        }

        obj.AddMember(
            rapidjson::StringRef("material"),
            jmat,
            allocator
            );
    }

    for (auto& t : sp->textures())
    {
        std::string channel_name = t.first;

        rapidjson::Value jtex;
        jtex.SetObject();

        if (auto tex = std::dynamic_pointer_cast<sg::texture2d<vector<4, unorm<8>>>>(t.second))
        {
            // Not implemented yet
        }
#if VSNRAY_COMMON_HAVE_PTEX
        else if (auto tex = std::dynamic_pointer_cast<sg::ptex_texture>(t.second))
        {
            jtex.AddMember(
                rapidjson::StringRef("type"),
                rapidjson::StringRef("ptex"),
                allocator
                );

            rapidjson::Value filename(tex->filename().c_str(), allocator);
            jtex.AddMember(
                rapidjson::StringRef("filename"),
                filename,
                allocator
                );
        }
#endif // VSNRAY_COMMON_HAVE_PTEX
        else
        {
            throw std::runtime_error("");
        }
    }
}

template <typename Object>
void vsnray_writer::write_triangle_mesh(Object obj, std::shared_ptr<sg::triangle_mesh> const& tm)
{
    auto& allocator = document_.GetAllocator();

    // Write binary files

    if (!tm->vertices.empty())
    {
        data_file::meta_data md;
        md.path = make_inline_filename(tm->name(), "vert");
        md.encoding = data_file::meta_data::Binary;
        md.data_type = data_file::meta_data::Vec3f;
        md.num_items = tm->vertices.size();

        rapidjson::Value verts;
        verts.SetObject();

        write_data_file(verts.GetObject(), md, tm->vertices);

        obj.AddMember("vertices", verts, allocator);
    }

    if (!tm->normals.empty())
    {
        data_file::meta_data md;
        md.path = make_inline_filename(tm->name(), "norm");
        md.encoding = data_file::meta_data::Binary;
        md.data_type = data_file::meta_data::Vec3f;
        md.num_items = tm->normals.size();

        rapidjson::Value norms;
        norms.SetObject();

        write_data_file(norms.GetObject(), md, tm->normals);

        obj.AddMember("normals", norms, allocator);
    }

    if (!tm->tex_coords.empty())
    {
        data_file::meta_data md;
        md.path = make_inline_filename(tm->name(), "texc");
        md.encoding = data_file::meta_data::Binary;
        md.data_type = data_file::meta_data::Vec2f;
        md.num_items = tm->tex_coords.size();

        rapidjson::Value tcs;
        tcs.SetObject();

        write_data_file(tcs.GetObject(), md, tm->tex_coords);

        obj.AddMember("tex_coords", tcs, allocator);
    }

    if (!tm->colors.empty())
    {
        data_file::meta_data md;
        md.path = make_inline_filename(tm->name(), "colo");
        md.encoding = data_file::meta_data::Binary;
        md.data_type = data_file::meta_data::Vec3u8;
        md.num_items = tm->colors.size();

        rapidjson::Value cols;
        cols.SetObject();

        write_data_file(cols.GetObject(), md, tm->colors);

        obj.AddMember("colors", cols, allocator);
    }
}

template <typename Object>
void vsnray_writer::write_indexed_triangle_mesh(Object obj, std::shared_ptr<sg::indexed_triangle_mesh> const& itm)
{
}

template <typename Object, typename Container>
void vsnray_writer::write_data_file(Object obj, data_file::meta_data md, Container const& cont)
{
    // First try to write the actual data file

    // Don't overwrite
    if (boost::filesystem::exists(md.path))
    {
        throw std::runtime_error("");
    }

    // Check for consistency
    if (static_cast<int>(cont.size()) != md.num_items)
    {
        throw std::runtime_error("");
    }

    std::ofstream file(md.path, std::ios::binary);

    if (!file.good())
    {
        throw std::runtime_error("");
    }

    // Write data
    try
    {
        file.write(reinterpret_cast<char const*>(cont.data()), cont.size() * sizeof(typename Container::value_type));
    }
    catch (std::ios_base::failure)
    {
        throw std::runtime_error("");
    }

    assert(boost::filesystem::exists(md.path));


    // Now store a JSON node containing meta data to the document
    auto& allocator = document_.GetAllocator();

    std::string data_type = "";

    auto& data_type_map = data_file::meta_data::data_type_map;

    auto it = data_type_map.left.find(md.data_type);

    if (it != data_type_map.left.end())
    {
        data_type = it->second;
    }
    else
    {
        throw std::runtime_error("");
    }

    assert(!data_type.empty());

    if (md.encoding == data_file::meta_data::Binary)
    {
        obj.AddMember(
            rapidjson::StringRef("type"),
            rapidjson::StringRef("file"),
            allocator
            );

        rapidjson::Value path(md.path.c_str(), allocator);
        obj.AddMember(
            rapidjson::StringRef("path"),
            path,
            allocator
            );

        obj.AddMember(
            rapidjson::StringRef("encoding"),
            rapidjson::StringRef("binary"),
            allocator
            );

        rapidjson::Value dt(data_type.c_str(), allocator);
        obj.AddMember(
            rapidjson::StringRef("data_type"),
            dt,
            allocator
            );

        rapidjson::Value num_items;
        num_items.SetInt(md.num_items);
        obj.AddMember("num_items", num_items, allocator);

        obj.AddMember(
            rapidjson::StringRef("compression"),
            rapidjson::StringRef("none"),
            allocator
            );
    }
    else
    {
        // Not implemented yet
        throw std::runtime_error("");
    }
}

std::string vsnray_writer::make_inline_filename(std::string node_name, std::string suffix)
{
    std::string result;

    std::string insert = node_name;

    int inc = 0;

    for (;;)
    {
        std::string fn = filename_;

        if (!node_name.empty())
        {
            fn.append(std::string(".") + node_name);
        }
        else
        {
            std::string inc_str = std::to_string(inc);

            while (inc_str.length() < 8)
            {
                inc_str = std::string("0") + inc_str;
            }

            fn.append(std::string(".") + inc_str);

            ++inc;
        }

        if (!suffix.empty())
        {
            if (suffix[0] != '.')
            {
                fn.append(".");
            }

            fn.append(suffix);
        }

        if (!boost::filesystem::exists(fn))
        {
            result = fn;
            break;
        }
        else
        {
            if (!node_name.empty())
            {
                throw std::runtime_error("");
            }
        }
    }

    return result;
}


namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Interface
//

void load_vsnray(std::string const& filename, model& mod)
{
    std::vector<std::string> filenames(1);

    filenames[0] = filename;

    load_vsnray(filenames, mod);
}

void save_vsnray(std::string const& filename, model const& mod, file_base::save_options const& options)
{
    cfile file(filename, "w+");
    if (!file.good())
    {
        std::cerr << "Cannot open " << filename << '\n';
        return;
    }

    rapidjson::Document doc;
    doc.SetObject();

    vsnray_writer writer(doc, filename);
    writer.write_node(doc.GetObject(), mod.scene_graph);

    char buffer[65536];
    rapidjson::FileWriteStream fws(file.get(), buffer, sizeof(buffer));

    rapidjson::PrettyWriter<rapidjson::FileWriteStream> w(fws);
    doc.Accept(w);
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
        mod.scene_graph = root;
    }
    else
    {
        mod.scene_graph->add_child(root);
    }
}

} // visionaray
