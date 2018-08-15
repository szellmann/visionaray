// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
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
            normals->t == tinyply::Type::FLOAT32 &&
            faces->t == tinyply::Type::INT32
            );

        float const* verts = reinterpret_cast<float const*>(vertices->buffer.get());
        int const* indices = reinterpret_cast<int const*>(faces->buffer.get());
        float const* norms = normals ? reinterpret_cast<float const*>(normals->buffer.get()) : nullptr;
        float const* coords = normals ? reinterpret_cast<float const*>(tex_coords->buffer.get()) : nullptr;

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

            mod.geometric_normals.emplace_back(cross(e1, e2));

            mod.tex_coords.emplace_back(0.0f);
            mod.tex_coords.emplace_back(0.0f);
            mod.tex_coords.emplace_back(0.0f);

            mod.bbox.insert(v1);
            mod.bbox.insert(v2);
            mod.bbox.insert(v3);
        }

        mod.materials.emplace_back(model::material_type());
        mod.textures.push_back({0, 0});
    }
    catch (std::exception const& e)
    {
        std::cerr << "Tinyply exception: " << e.what() << '\n';
    }
}

} // visionaray
