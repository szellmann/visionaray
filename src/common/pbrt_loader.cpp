// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <algorithm>
#include <cctype>
#include <cstring> // memcpy
#include <stdexcept>
#include <unordered_map>

#include <boost/filesystem.hpp>

#if VSNRAY_COMMON_HAVE_PBRTPARSER
#include <pbrtParser/Scene.h>
#endif

#include <visionaray/detail/spd/blackbody.h>
#include <visionaray/math/forward.h>
#include <visionaray/math/io.h>
#include <visionaray/math/matrix.h>
#include <visionaray/math/vector.h>

#include "image.h"
#include "make_texture.h"
#include "model.h"
#include "sg.h"

namespace visionaray
{

#if VSNRAY_COMMON_HAVE_PBRTPARSER

using namespace pbrt;

void add_diffuse_texture(std::shared_ptr<sg::surface_properties>& sp, Texture::SP texture)
{
    if (auto t = std::dynamic_pointer_cast<ImageTexture>(texture))
    {
        image img;
        if (img.load(t->fileName))
        {
            auto tex = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>(make_texture(img));

            sp->add_texture(tex, "diffuse");
        }
    }
    else if (auto t = std::dynamic_pointer_cast<ConstantTexture>(texture))
    {
        vector<4, unorm<8>> texel(t->value.x, t->value.y, t->value.z, 1.0f);

        auto tex = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>();
        tex->resize(1, 1);
        tex->set_address_mode(Wrap);
        tex->set_filter_mode(Nearest);
        tex->reset(&texel);

        sp->add_texture(tex, "diffuse");
    }
}

mat4 make_mat4(affine3f const& aff)
{
    mat4 result = mat4::identity();

    result(0, 0) = aff.l.vx.x;
    result(1, 0) = aff.l.vx.y;
    result(2, 0) = aff.l.vx.z;

    result(0, 1) = aff.l.vy.x;
    result(1, 1) = aff.l.vy.y;
    result(2, 1) = aff.l.vy.z;

    result(0, 2) = aff.l.vz.x;
    result(1, 2) = aff.l.vz.y;
    result(2, 2) = aff.l.vz.z;

    result(0, 3) = aff.p.x;
    result(1, 3) = aff.p.y;
    result(2, 3) = aff.p.z;

    return result;
}

std::shared_ptr<sg::surface_properties> make_surface_properties(Shape::SP shape)
{
    auto sp = std::make_shared<sg::surface_properties>();

    if (auto m = std::dynamic_pointer_cast<MetalMaterial>(shape->material))
    {
    }
    else if (auto m = std::dynamic_pointer_cast<PlasticMaterial>(shape->material))
    {
        auto obj = std::make_shared<sg::obj_material>();

        obj->cd = vec3(m->kd.x, m->kd.y, m->kd.z);
        obj->cs = vec3(m->ks.x, m->ks.y, m->ks.z);

        // TODO
        obj->specular_exp = m->roughness;

        sp->material() = obj;

        add_diffuse_texture(sp, m->map_kd);
    }
    else if (auto m = std::dynamic_pointer_cast<SubstrateMaterial>(shape->material))
    {
        auto obj = std::make_shared<sg::obj_material>();

        obj->cd = vec3(m->kd.x, m->kd.y, m->kd.z);
        obj->cs = vec3(m->ks.x, m->ks.y, m->ks.z);

        // TODO
        obj->specular_exp = m->uRoughness;

        sp->material() = obj;

        add_diffuse_texture(sp, m->map_kd);
    }
    else if (auto m = std::dynamic_pointer_cast<MirrorMaterial>(shape->material))
    {
        auto obj = std::make_shared<sg::obj_material>();

        obj->illum = 3; // indicates purely reflective!

        obj->cs = vec3(m->kr.x, m->kr.y, m->kr.z);

        sp->material() = obj;
    }
    else if (auto m = std::dynamic_pointer_cast<MatteMaterial>(shape->material))
    {
        auto obj = std::make_shared<sg::obj_material>();

        obj->cd = vec3(m->kd.x, m->kd.y, m->kd.z);

        sp->material() = obj;

        add_diffuse_texture(sp, m->map_kd);
    }
    else if (auto m = std::dynamic_pointer_cast<GlassMaterial>(shape->material))
    {
        auto glass = std::make_shared<sg::glass_material>();

        glass->ct = vec3(m->kt.x, m->kt.y, m->kt.z);
        glass->cr = vec3(m->kr.x, m->kr.y, m->kr.z);
        glass->ior = vec3(m->index);

        sp->material() = glass;
    }
    else if (auto m = std::dynamic_pointer_cast<UberMaterial>(shape->material))
    {
        auto obj = std::make_shared<sg::obj_material>();

        obj->cd = vec3(m->kd.x, m->kd.y, m->kd.z);
        obj->cs = vec3(m->ks.x, m->ks.y, m->ks.z);

        sp->material() = obj;

        add_diffuse_texture(sp, m->map_kd);
    }

    return sp;
}

void make_scene_graph(
        Object::SP object,
        sg::node& parent,
        std::unordered_map<Shape::SP, std::shared_ptr<sg::indexed_triangle_mesh>>& shape2itm,
        std::unordered_map<Material::SP, std::shared_ptr<sg::surface_properties>>& mat2prop
        )
{
    for (auto shape : object->shapes)
    {
        if (auto sphere = std::dynamic_pointer_cast<Sphere>(shape))
        {
            mat4 m = mat4::identity();
            m = translate(m, vec3(sphere->transform.p.x, sphere->transform.p.y, sphere->transform.p.z));
            m = scale(m, vec3(sphere->radius));

            auto trans = std::make_shared<sg::transform>();
            trans->matrix() = m;

            trans->add_child(std::make_shared<sg::sphere>());

            std::shared_ptr<sg::surface_properties> sp = nullptr;

            if (shape->areaLight != nullptr)
            {
                // If shape has an area light, ignore material!

                sp = std::make_shared<sg::surface_properties>();

                // TODO: don't misuse obj for emissive
                auto obj = std::make_shared<sg::obj_material>();

                if (auto l = std::dynamic_pointer_cast<DiffuseAreaLightRGB>(shape->areaLight))
                {
                    obj->ce = vec3(l->L.x, l->L.y, l->L.z);
                }
                else if (auto l = std::dynamic_pointer_cast<DiffuseAreaLightBB>(shape->areaLight))
                {
                    blackbody bb(l->temperature);

                    obj->ce = spd_to_rgb(bb) * l->scale; //?
                }

                sp->material() = obj;
            }
            else
            {
                auto it = mat2prop.find(sphere->material);

                if (it != mat2prop.end())
                {
                    sp = std::make_shared<sg::surface_properties>();
                    sp->material() = it->second->material();
                    sp->textures() = it->second->textures();
                }
                else
                {
                    sp = make_surface_properties(sphere);
                    mat2prop.insert({ sphere->material, sp });
                }
            }

            parent.add_child(sp);

            bool insert_dummy = sp->textures().size() == 0;

            if (insert_dummy)
            {
                // Add a dummy texture
                vector<4, unorm<8>> dummy_texel(1.0f, 1.0f, 1.0f, 1.0f);
                auto tex = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>();
                tex->resize(1, 1);
                tex->set_address_mode(Wrap);
                tex->set_filter_mode(Nearest);
                tex->reset(&dummy_texel);
                sp->add_texture(tex, "diffuse");
            }

            sp->add_child(trans);
        }

        if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(shape))
        {
            std::shared_ptr<sg::indexed_triangle_mesh> itm = nullptr;
            auto itm_it = shape2itm.find(shape);

            if (itm_it != shape2itm.end())
            {
                itm = itm_it->second;
            }
            else
            {
                itm = std::make_shared<sg::indexed_triangle_mesh>();

                itm->vertices = std::make_shared<aligned_vector<vec3>>(mesh->vertex.size());

                if (mesh->normal.size() > 0)
                {
                    itm->normals = std::make_shared<aligned_vector<vec3>>(mesh->normal.size());
                }

                if (mesh->texcoord.size() > 0)
                {
                    itm->tex_coords = std::make_shared<aligned_vector<vec2>>(mesh->texcoord.size());
                }

                itm->vertex_indices.resize(mesh->index.size() * 3);
                itm->normal_indices.resize(mesh->index.size() * 3);
                itm->tex_coord_indices.resize(mesh->index.size() * 3);

                for (size_t i = 0; i < mesh->vertex.size(); ++i)
                {
                    auto v = mesh->vertex[i];
                    (*itm->vertices)[i] = vec3(v.x, v.y, v.z);
                }

                for (size_t i = 0; i < mesh->normal.size(); ++i)
                {
                    auto n = mesh->normal[i];
                    (*itm->normals)[i] = vec3(n.x, n.y, n.z);
                }

                for (size_t i = 0; i < mesh->texcoord.size(); ++i)
                {
                    auto tc = mesh->texcoord[i];
                    (*itm->tex_coords)[i] = vec2(tc.x, tc.y);
                }

                for (size_t i = 0; i < mesh->index.size(); ++i)
                {
                    auto i3 = mesh->index[i];
                    memcpy(itm->vertex_indices.data() + i * 3, &i3.x, sizeof(int) * 3);
                    memcpy(itm->normal_indices.data() + i * 3, &i3.x, sizeof(int) * 3);
                    memcpy(itm->tex_coord_indices.data() + i * 3, &i3.x, sizeof(int) * 3);
                }

                // If model has no shading normals, use geometric normals instead
                if (itm->normals == nullptr)
                {
                    itm->normals = std::make_shared<aligned_vector<vec3>>(itm->vertex_indices.size());

                    for (size_t i = 0; i < itm->vertex_indices.size(); i += 3)
                    {
                        int i1 = itm->vertex_indices[i];
                        int i2 = itm->vertex_indices[i + 1];
                        int i3 = itm->vertex_indices[i + 2];

                        vec3 v1 = (*itm->vertices)[i1];
                        vec3 v2 = (*itm->vertices)[i2];
                        vec3 v3 = (*itm->vertices)[i3];

                        vec3 n = normalize(cross(v2 - v1, v3 - v1));

                        (*itm->normals)[i1] = n;
                        (*itm->normals)[i2] = n;
                        (*itm->normals)[i3] = n;
                    }
                }

                // If model has no texture coordinates, add dummies
                if (itm->tex_coords == nullptr)
                {
                    itm->tex_coords = std::make_shared<aligned_vector<vec2>>(itm->tex_coord_indices.size());

                    std::fill(itm->tex_coords->begin(), itm->tex_coords->end(), vec2(0.0f, 0.0f));
                }

                shape2itm.insert({ shape, itm });
            }

            std::shared_ptr<sg::surface_properties> sp = nullptr;

            if (shape->areaLight != nullptr)
            {
                // If shape has an area light, ignore material!

                sp = std::make_shared<sg::surface_properties>();

                // TODO: don't misuse obj for emissive
                auto obj = std::make_shared<sg::obj_material>();

                if (auto l = std::dynamic_pointer_cast<DiffuseAreaLightRGB>(shape->areaLight))
                {
                    obj->ce = vec3(l->L.x, l->L.y, l->L.z);
                }
                else if (auto l = std::dynamic_pointer_cast<DiffuseAreaLightBB>(shape->areaLight))
                {
                    blackbody bb(l->temperature);

                    obj->ce = spd_to_rgb(bb) * l->scale; //?
                }

                sp->material() = obj;
            }
            else
            {
                auto it = mat2prop.find(mesh->material);

                if (it != mat2prop.end())
                {
                    sp = std::make_shared<sg::surface_properties>();
                    sp->material() = it->second->material();
                    sp->textures() = it->second->textures();
                }
                else
                {
                    sp = make_surface_properties(mesh);
                    mat2prop.insert({ mesh->material, sp });
                }
            }

            parent.add_child(sp);

            bool insert_dummy = sp->textures().size() == 0;

            if (insert_dummy)
            {
                // Add a dummy texture
                vector<4, unorm<8>> dummy_texel(1.0f, 1.0f, 1.0f, 1.0f);
                auto tex = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>();
                tex->resize(1, 1);
                tex->set_address_mode(Wrap);
                tex->set_filter_mode(Nearest);
                tex->reset(&dummy_texel);
                sp->add_texture(tex, "diffuse");
            }

            sp->add_child(itm);
        }
    }

    for (auto inst : object->instances)
    {
        auto trans = std::make_shared<sg::transform>();

        trans->matrix() = make_mat4(inst->xfm);

        make_scene_graph(inst->object, *trans, shape2itm, mat2prop);

        parent.add_child(trans);
    }
}

#endif // VSNRAY_COMMON_HAVE_PBRTPARSER

void load_pbrt(std::string const& filename, model& mod)
{
#if VSNRAY_COMMON_HAVE_PBRTPARSER
    auto root = std::make_shared<sg::node>();

    // If we find a material that is already in use, we just hang
    // the triangle mesh underneath its surface_properties node
    // NOTE: this only works because pbrtParser doesn't expose
    // transforms
    std::unordered_map<Material::SP, std::shared_ptr<sg::surface_properties>> mat2prop;

    // List with all shapes that are stored and their corresponding
    // triangle mesh nodes, so that instances can refer to them
    std::unordered_map<Shape::SP, std::shared_ptr<sg::indexed_triangle_mesh>> shape2itm;

    std::shared_ptr<Scene> scene;

    try
    {
        boost::filesystem::path p(filename);
        std::string ext = p.extension().string();

        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".pbf")
        {
            scene = Scene::loadFrom(filename);
        }
        else if (ext == ".pbrt")
        {
            scene = importPBRT(filename);
        }

        make_scene_graph(scene->world, *root, shape2itm, mat2prop);
    }
    catch (std::runtime_error e)
    {
        // TODO
        throw e;
    }

    if (mod.scene_graph == nullptr)
    {
        mod.scene_graph = root;
    }
    else
    {
        mod.scene_graph->add_child(root);
    }
#else
    VSNRAY_UNUSED(filename, mod);
#endif
}

void load_pbrt(std::vector<std::string> const& filenames, model& mod)
{
    // TODO!
    for (auto filename : filenames)
    {
        load_pbrt(filename, mod);
    }
}

} // visionaray
