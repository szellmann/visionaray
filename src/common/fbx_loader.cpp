// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <algorithm>
#include <cstdint>
#include <cstring> // memcpy
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#if VSNRAY_COMMON_HAVE_ZLIB
#include <zlib.h>
#endif

#include "fbx_loader.h"
#include "make_unique.h"
#include "model.h"
#include "sg.h"

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Helpers
//

#if VSNRAY_COMMON_HAVE_ZLIB

int unzip(Bytef* dst, unsigned dst_len, Bytef const* src, unsigned src_len)
{
    z_stream zstream = {};

    zstream.next_out  = dst;
    zstream.avail_out = dst_len;
    zstream.total_out = dst_len;

    zstream.next_in   = const_cast<Bytef*>(src);
    zstream.avail_in  = src_len;
    zstream.total_in  = src_len;

    int err = inflateInit(&zstream);

    if (err != Z_OK)
    {
        return err;
    }

    for (;;)
    {
        err = inflate(&zstream, Z_FINISH);
        if (err == Z_STREAM_END)
        {
            break;
        }

        if (err != Z_OK)
        {
            return err;
        }
    }

    err = inflateEnd(&zstream);

    return err;
}

#endif

template <typename Container>
void parse_fbx_property_array(Container& cont, char* data, size_t& offset)
{
    uint32_t len = 0;
    std::memcpy(&len, data + offset, sizeof(len));
    offset += sizeof(len);

    uint32_t encoding = 0;
    std::memcpy(&encoding, data + offset, sizeof(encoding));
    offset += sizeof(encoding);

    uint32_t compressed_len = 0;
    std::memcpy(&compressed_len, data + offset, sizeof(compressed_len));
    offset += sizeof(compressed_len);

    cont.resize(len);

    if (encoding)
    {
#if VSNRAY_COMMON_HAVE_ZLIB
        int res = unzip(
            reinterpret_cast<Bytef*>(cont.data()),
            len * sizeof(typename Container::value_type),
            reinterpret_cast<Bytef const*>(data + offset),
            compressed_len
            );

        // TODO: error handling
        switch (res)
        {
        case Z_OK:
            break;

        case Z_MEM_ERROR:
            break;

        case Z_BUF_ERROR:
            break;

        case Z_DATA_ERROR:
            break;

        default:
            break;
        }
#else
        // TODO: error handling
#endif
        offset += compressed_len;
    }
    else
    {
        offset += len * sizeof(typename Container::value_type);
    }
}


//-------------------------------------------------------------------------------------------------
// fbx properties
//

struct fbx_property
{
    virtual ~fbx_property() {}
};

struct fbx_property_string : fbx_property, std::string
{
};

struct fbx_property_raw : fbx_property, std::vector<char>
{
};

struct fbx_property_int32s : fbx_property, std::vector<int32_t>
{
};

struct fbx_property_int64s : fbx_property, std::vector<int64_t>
{
};

struct fbx_property_floats : fbx_property, std::vector<float>
{
};

struct fbx_property_doubles : fbx_property, std::vector<double>
{
};


//-------------------------------------------------------------------------------------------------
// fbx_node
//

struct fbx_node
{
    using properties = std::vector<std::unique_ptr<fbx_property>>;

    std::string name;
    uint32_t end_offset;

    properties props;
    std::vector<std::unique_ptr<fbx_node>> children;
};


//-------------------------------------------------------------------------------------------------
// fbx_document
//

class fbx_document
{
public:

    unsigned get_version() const;

    std::shared_ptr<sg::node> parse(std::ifstream& stream);

private:

    // Parse nodes
    std::unique_ptr<fbx_node> parse_fbx_node(std::ifstream& stream);

    // Parse properties
    fbx_node::properties parse_fbx_properties(char* data, size_t len);

    std::unique_ptr<fbx_property_raw>     parse_fbx_property_raw(char* data, size_t& offset);
    std::unique_ptr<fbx_property_string>  parse_fbx_property_string(char* data, size_t& offset);
    std::unique_ptr<fbx_property_int32s>  parse_fbx_property_int32s(char* data, size_t& offset);
    std::unique_ptr<fbx_property_int64s>  parse_fbx_property_int64s(char* data, size_t& offset);
    std::unique_ptr<fbx_property_floats>  parse_fbx_property_floats(char* data, size_t& offset);
    std::unique_ptr<fbx_property_doubles> parse_fbx_property_doubles(char* data, size_t& offset);

    // Process nodes
    void process_node(std::shared_ptr<sg::node> dst, std::unique_ptr<fbx_node> const& src);

    unsigned version_;

    // Top level nodes
    std::vector<std::unique_ptr<fbx_node>> fbx_nodes_;
};


unsigned fbx_document::get_version() const
{
    return version_;
}

std::shared_ptr<sg::node> fbx_document::parse(std::ifstream& stream)
{
    char header[27];

    stream.read(header, 27);

    if (header[21] != 0x1A || header[22] != 0x00)
    {
        return nullptr;
    }

    std::memcpy(&version_, header + 23, 4);

    auto root = std::make_shared<sg::node>();

    auto node = parse_fbx_node(stream);

    while (node != nullptr && node->end_offset > 0)
    {
        stream.seekg(node->end_offset);

        fbx_nodes_.push_back(std::move(node));

        node = parse_fbx_node(stream);
    }

    for (auto const& n : fbx_nodes_)
    {
        process_node(root, n);
    }

    return root;
}

std::unique_ptr<fbx_node> fbx_document::parse_fbx_node(std::ifstream& stream)
{
    if (stream.eof())
    {
        return nullptr;
    }

    auto node = make_unique<fbx_node>();

    uint32_t num_properties;
    uint32_t property_list_len;
    uint8_t  name_len;

    stream.read(reinterpret_cast<char*>(&node->end_offset), sizeof(node->end_offset));
    stream.read(reinterpret_cast<char*>(&num_properties), sizeof(num_properties));
    stream.read(reinterpret_cast<char*>(&property_list_len), sizeof(property_list_len));
    stream.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

    std::vector<char> name_bytes(name_len);
    stream.read(name_bytes.data(), name_len);

    node->name = std::string(name_bytes.data(), name_len);

    // Parse node properties
    std::vector<char> property_list_bytes(property_list_len);
    stream.read(property_list_bytes.data(), property_list_len);

    if (num_properties != 0)
    {
        node->props = parse_fbx_properties(property_list_bytes.data(), property_list_bytes.size());
    }

    // Parse nested nodes
    if (stream.tellg() < node->end_offset)
    {
        auto child = parse_fbx_node(stream);

        while (child != nullptr && stream.tellg() < node->end_offset)
        {
            stream.seekg(child->end_offset);

            node->children.push_back(std::move(child));

            child = parse_fbx_node(stream);
        }
    }

    return node;
}

fbx_node::properties fbx_document::parse_fbx_properties(char* data, size_t len)
{
    fbx_node::properties result;

    size_t off = 0;

    while (off < len)
    {
        char type = data[off];
        off += 1;

        switch (type)
        {
        case 'R': // raw data
            result.push_back(parse_fbx_property_raw(data, off));
            break;

        case 'S': // string
            result.push_back(parse_fbx_property_string(data, off));
            break;

        case 'i': // array of 32-bit ints
            result.push_back(parse_fbx_property_int32s(data, off));
            break;

        case 'l': // array of 64-bit ints
            result.push_back(parse_fbx_property_int64s(data, off));
            break;

        case 'f': // array of floats
            result.push_back(parse_fbx_property_floats(data, off));
            break;

        case 'd': // array of doubles
            result.push_back(parse_fbx_property_doubles(data, off));
            break;

        default:
            // Just break out of loop!
            // TODO: error handling
            off = len;
            break;
        }
    }

    return result;
}

std::unique_ptr<fbx_property_raw> fbx_document::parse_fbx_property_raw(char* data, size_t& offset)
{
    auto result = make_unique<fbx_property_raw>();

    uint32_t len = 0;
    std::memcpy(&len, data + offset, sizeof(len));
    offset += sizeof(len);

    result->resize(len);
    std::memcpy(result->data(), data + offset, result->size() * sizeof(char));
    offset += result->size() * sizeof(char);
    
    return result;
}

std::unique_ptr<fbx_property_string> fbx_document::parse_fbx_property_string(char* data, size_t& offset)
{
    auto result = make_unique<fbx_property_string>();

    uint32_t len = 0;
    std::memcpy(&len, data + offset, sizeof(len));
    offset += sizeof(len);
    
    std::vector<char> chars(len);
    std::memcpy(chars.data(), data + offset, chars.size() * sizeof(char));
    offset += chars.size() * sizeof(char);

    result->assign(chars.data(), chars.size());

    return result;
}

std::unique_ptr<fbx_property_int32s> fbx_document::parse_fbx_property_int32s(char* data, size_t& offset)
{
    auto result = make_unique<fbx_property_int32s>();

    parse_fbx_property_array(*result, data, offset);

    return result;
}

std::unique_ptr<fbx_property_int64s> fbx_document::parse_fbx_property_int64s(char* data, size_t& offset)
{
    auto result = make_unique<fbx_property_int64s>();

    parse_fbx_property_array(*result, data, offset);

    return result;
}

std::unique_ptr<fbx_property_floats> fbx_document::parse_fbx_property_floats(char* data, size_t& offset)
{
    auto result = make_unique<fbx_property_floats>();

    parse_fbx_property_array(*result, data, offset);

    return result;
}

std::unique_ptr<fbx_property_doubles> fbx_document::parse_fbx_property_doubles(char* data, size_t& offset)
{
    auto result = make_unique<fbx_property_doubles>();

    parse_fbx_property_array(*result, data, offset);

    return result;
}

void fbx_document::process_node(std::shared_ptr<sg::node> dst, std::unique_ptr<fbx_node> const& src)
{
    if (src->name == "Geometry")
    {
        auto mesh = std::make_shared<sg::indexed_triangle_mesh>();
        dst->add_child(mesh);
        dst = mesh;
    }
    else if (src->name == "Vertices")
    {
        auto mesh = std::dynamic_pointer_cast<sg::indexed_triangle_mesh>(dst);

        if (dst == nullptr)
        {
            // TODO: error handling
        }

        if (src->props.size() != 1)
        {
            // TODO: error handling
        }

        if (auto vertices = dynamic_cast<fbx_property_floats const*>(src->props[0].get()))
        {
            mesh->vertices.resize(vertices->size() / 3);

            for (size_t i = 0; i < vertices->size(); i += 3)
            {
                mesh->vertices[i / 3] = vec3(
                        (*vertices)[i],
                        (*vertices)[i + 1],
                        (*vertices)[i + 2]
                        );

            }
        }
        else if (auto vertices = dynamic_cast<fbx_property_doubles const*>(src->props[0].get()))
        {
            mesh->vertices.resize(vertices->size() / 3);

            for (size_t i = 0; i < vertices->size(); i += 3)
            {
                // Convert double to float
                mesh->vertices[i / 3] = vec3(
                        static_cast<float>((*vertices)[i]),
                        static_cast<float>((*vertices)[i + 1]),
                        static_cast<float>((*vertices)[i + 2])
                        );

            }
        }
        else
        {
            // TODO: error handling
        }
    }
    else if (src->name == "Normals")
    {
        auto mesh = std::dynamic_pointer_cast<sg::indexed_triangle_mesh>(dst);

        if (dst == nullptr)
        {
            // TODO: error handling
        }

        if (src->props.size() != 1)
        {
            // TODO: error handling
        }

        if (auto normals = dynamic_cast<fbx_property_floats const*>(src->props[0].get()))
        {
            mesh->normals.resize(normals->size() / 3);

            for (size_t i = 0; i < normals->size(); i += 3)
            {
                mesh->normals[i / 3] = vec3(
                        (*normals)[i],
                        (*normals)[i + 1],
                        (*normals)[i + 2]
                        );

            }
        }
        else if (auto normals = dynamic_cast<fbx_property_doubles const*>(src->props[0].get()))
        {
            mesh->normals.resize(normals->size() / 3);

            for (size_t i = 0; i < normals->size(); i += 3)
            {
                // Convert double to float
                mesh->normals[i / 3] = vec3(
                        static_cast<float>((*normals)[i]),
                        static_cast<float>((*normals)[i + 1]),
                        static_cast<float>((*normals)[i + 2])
                        );

            }
        }
        else
        {
            // TODO: error handling
        }
    }
    else if (src->name == "PolygonVertexIndex")
    {
        auto mesh = std::dynamic_pointer_cast<sg::indexed_triangle_mesh>(dst);

        if (dst == nullptr)
        {
            // TODO: error handling
        }

        if (src->props.size() != 1)
        {
            // TODO: error handling
        }

        if (auto indices = dynamic_cast<fbx_property_int32s const*>(src->props[0].get()))
        {
            mesh->indices.resize(indices->size());

            for (size_t i = 0; i < indices->size(); ++i)
            {
                int32_t index = (*indices)[i];

                if (index < 0)
                {
                    index = ~index;
                }

                mesh->indices[i] = index;
            }
        }
        else
        {
            // TODO: error handling
        }
    }
    else if (src->name == "UV")
    {
        auto mesh = std::dynamic_pointer_cast<sg::indexed_triangle_mesh>(dst);

        if (dst == nullptr)
        {
            // TODO: error handling
        }

        if (src->props.size() != 1)
        {
            // TODO: error handling
        }

        if (auto uvs = dynamic_cast<fbx_property_floats const*>(src->props[0].get()))
        {
            mesh->tex_coords.resize(uvs->size() / 2);

            for (size_t i = 0; i < uvs->size(); i += 2)
            {
                mesh->tex_coords[i / 2] = vec2(
                        (*uvs)[i],
                        (*uvs)[i + 1]
                        );

            }
        }
        else if (auto uvs = dynamic_cast<fbx_property_doubles const*>(src->props[0].get()))
        {
            mesh->tex_coords.resize(uvs->size() / 2);

            for (size_t i = 0; i < uvs->size(); i += 2)
            {
                // Convert double to float
                mesh->tex_coords[i / 2] = vec2(
                        static_cast<float>((*uvs)[i]),
                        static_cast<float>((*uvs)[i + 1])
                        );

            }
        }
        else
        {
            // TODO: error handling
        }
    }

    for (auto const& c : src->children)
    {
        process_node(dst, c);
    }
}


namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Interface
//

void load_fbx(std::string const& filename, model& mod)
{
    std::ifstream stream(filename, std::ios::binary);
    if (stream.fail())
    {
        std::cerr << "Cannot open " << filename << '\n';
        return;
    }

    if (mod.scene_graph == nullptr)
    {
        mod.scene_graph = std::make_shared<sg::node>();
    }

    fbx_document doc;
    auto root = doc.parse(stream);

    if (root != nullptr)
    {
        mod.scene_graph->add_child(root);
    }
    else
    {
        std::cerr << "Error loading " << filename << '\n';
    }
}

void load_fbx(std::vector<std::string> const& filenames, model& mod)
{
    for (auto filename : filenames)
    {
        load_fbx(filename, mod);
    }
}

} // visionaray
