// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>

#include <tinyply.h>

#include "ply_loader.h"
#include "model.h"

using namespace tinyply;

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Wrapper around tinyply index buffer (resolves index type)
//

class index_buffer
{
public:
    index_buffer(Buffer& buffer, Type type)
        : buffer_(buffer)
        , type_(type)
    {
    }

    int operator[](size_t index)
    {
        switch (type_)
        {
        case Type::INT8:
            return reinterpret_cast<int8_t const*>(buffer_.get())[index];
        case Type::UINT8:
            return reinterpret_cast<uint8_t const*>(buffer_.get())[index];
        case Type::INT16:
            return reinterpret_cast<int16_t const*>(buffer_.get())[index];
        case Type::UINT16:
            return reinterpret_cast<uint16_t const*>(buffer_.get())[index];
        case Type::INT32:
            return reinterpret_cast<int32_t const*>(buffer_.get())[index];
        case Type::UINT32:
            return reinterpret_cast<uint32_t const*>(buffer_.get())[index];
        default:
            assert(0);
        }

        return -1;
    }

private:
    Buffer& buffer_;
    Type type_;

};

void load_ply(std::string const& filename, model& mod)
{
	try
    {
        std::ifstream stream(filename, std::ios::binary);
        if (stream.fail())
        {
            std::cerr << "Cannot open " << filename << '\n';
            return;
        }

        PlyFile file;
        file.parse_header(stream);

        std::shared_ptr<PlyData> vertices;
        std::shared_ptr<PlyData> normals;
        std::shared_ptr<PlyData> faces;
        std::shared_ptr<PlyData> tex_coords;

        try
        {
            vertices = file.request_properties_from_element("vertex", { "x", "y", "z" });
            faces = file.request_properties_from_element("face", { "vertex_indices" });
        }
        catch (std::exception const& e)
        {
            std::cerr << "Tinyply exception: " << e.what() << '\n';
            return;
        }

        try
        {
            normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" });
            tex_coords = file.request_properties_from_element("vertex", { "u", "v" });
        }
        catch (std::exception const& e)
        {
        }

        file.read(stream);

        assert(
            vertices &&
            faces &&
            vertices->t == tinyply::Type::FLOAT32 &&
            normals->t == tinyply::Type::FLOAT32
            );

        float const* verts = reinterpret_cast<float const*>(vertices->buffer.get());
        float const* norms = normals ? reinterpret_cast<float const*>(normals->buffer.get()) : nullptr;
        float const* coords = tex_coords ? reinterpret_cast<float const*>(tex_coords->buffer.get()) : nullptr;

        index_buffer indices(faces->buffer, faces->t);

        if (mod.primitives.size() == 0)
        {
            mod.bbox.invalidate();
        }

        for (size_t i = 0; i < faces->count; ++i)
        {
            int index1 = indices[i * 3];
            int index2 = indices[i * 3 + 1];
            int index3 = indices[i * 3 + 2];

            vec3 v1(verts[index1 * 3], verts[index1 * 3 + 1], verts[index1 * 3 + 2]);
            vec3 v2(verts[index2 * 3], verts[index2 * 3 + 1], verts[index2 * 3 + 2]);
            vec3 v3(verts[index3 * 3], verts[index3 * 3 + 1], verts[index3 * 3 + 2]);
            vec3 e1 = v2 - v1;
            vec3 e2 = v3 - v1;

            basic_triangle<3, float> tri(v1, e1, e2);
            tri.prim_id = static_cast<unsigned>(mod.primitives.size());
            tri.geom_id = static_cast<unsigned>(mod.materials.size());

            mod.primitives.emplace_back(tri);

            if (norms)
            {
                vec3 n1(norms[index1 * 3], norms[index1 * 3 + 1], norms[index1 * 3 + 2]);
                vec3 n2(norms[index2 * 3], norms[index2 * 3 + 1], norms[index2 * 3 + 2]);
                vec3 n3(norms[index3 * 3], norms[index3 * 3 + 1], norms[index3 * 3 + 2]);

                mod.shading_normals.emplace_back(n1);
                mod.shading_normals.emplace_back(n2);
                mod.shading_normals.emplace_back(n3);
            }

            if (coords)
            {
                vec2 tc1(coords[index1 * 2], coords[index1 * 2 + 1]);
                vec2 tc2(coords[index2 * 2], coords[index2 * 2 + 1]);
                vec2 tc3(coords[index3 * 2], coords[index3 * 2 + 1]);

                mod.tex_coords.emplace_back(tc1);
                mod.tex_coords.emplace_back(tc2);
                mod.tex_coords.emplace_back(tc3);
            }
            else
            {
                mod.tex_coords.emplace_back(0.0f);
                mod.tex_coords.emplace_back(0.0f);
                mod.tex_coords.emplace_back(0.0f);
            }

            mod.geometric_normals.emplace_back(cross(e1, e2));

            mod.bbox.insert(v1);
            mod.bbox.insert(v2);
            mod.bbox.insert(v3);
        }

        mod.materials.emplace_back(model::material_type());

        bool insert_dummy = true;

        if (insert_dummy)
        {
            // Add a dummy texture
            vector<4, unorm<8>> dummy_texel(1.0f, 1.0f, 1.0f, 1.0f);
            model::texture_type tex(1, 1);
            tex.set_address_mode(Wrap);
            tex.set_filter_mode(Nearest);
            tex.reset(&dummy_texel);
            mod.textures.push_back(std::move(tex));
        }
        mod.textures.push_back({0, 0});
    }
    catch (std::exception const& e)
    {
        std::cerr << "Tinyply exception: " << e.what() << '\n';
    }
}

void load_ply(std::vector<std::string> const& filenames, model& mod)
{
    // TODO!
    for (auto filename : filenames)
    {
        load_ply(filename, mod);
    }
}

} // visionaray
