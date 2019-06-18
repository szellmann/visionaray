// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <algorithm>
#include <cctype>
#include <cstring> // memcpy
#include <stdexcept>
#include <unordered_map>

#include <boost/algorithm/string.hpp>
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

void add_diffuse_texture(
        std::shared_ptr<sg::surface_properties>& sp,
        Texture::SP texture,
        std::string base_filename
        )
{
    if (auto t = std::dynamic_pointer_cast<ImageTexture>(texture))
    {
        std::string tex_filename;

        boost::filesystem::path kdp(t->fileName);

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
            boost::filesystem::path p(base_filename);
            tex_filename = p.parent_path().string() + "/" + t->fileName;
            std::replace(tex_filename.begin(), tex_filename.end(), '\\', '/');
        }

        if (!boost::filesystem::exists(tex_filename))
        {
            boost::trim(tex_filename);
        }

        if (boost::filesystem::exists(tex_filename))
        {
            image img;
            if (img.load(tex_filename))
            {
                auto tex = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>();
                tex->resize(img.width(), img.height());
                tex->name() = t->fileName;
                make_texture(*tex, img);

                sp->add_texture(tex, "diffuse");
            }
        }
    }
    else if (auto t = std::dynamic_pointer_cast<ConstantTexture>(texture))
    {
        vector<4, unorm<8>> texel(t->value.x, t->value.y, t->value.z, 1.0f);

        static unsigned num = 1;
        std::string name = "ConstantTexture" + std::to_string(num++);

        auto tex = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>();
        tex->resize(1, 1);
        tex->name() = name;
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

std::shared_ptr<sg::surface_properties> make_surface_properties(Shape::SP shape, std::string base_filename)
{
    auto sp = std::make_shared<sg::surface_properties>();

    if (auto m = std::dynamic_pointer_cast<DisneyMaterial>(shape->material))
    {
        auto disney = std::make_shared<sg::disney_material>();

        disney->base_color = vec4(m->color.x, m->color.y, m->color.z, 1.0f);
        disney->spec_trans = m->specTrans;
        disney->sheen = m->sheen;
        disney->sheen_tint = m->sheenTint;
        disney->ior = m->eta;
        disney->refractive = 0.0f;
        disney->roughness = m->roughness;

        sp->material() = disney;
    }
    else if (auto m = std::dynamic_pointer_cast<MetalMaterial>(shape->material))
    {
        auto obj = std::make_shared<sg::obj_material>();

        obj->name() = m->name;

        // Approximate with an obj material with no diffuse and a high exponent for now..
        obj->ca = vec3(0.0f, 0.0f, 0.0f);
        obj->cd = vec3(0.0f, 0.0f, 0.0f);
        obj->cs = vec3(m->k.x, m->k.y, m->k.z);
        obj->specular_exp = 128.0f;

        sp->material() = obj;
    }
    else if (auto m = std::dynamic_pointer_cast<PlasticMaterial>(shape->material))
    {
        auto obj = std::make_shared<sg::obj_material>();
        
        obj->name() = m->name;

        obj->cd = vec3(m->kd.x, m->kd.y, m->kd.z);
        obj->cs = vec3(m->ks.x, m->ks.y, m->ks.z);

        // TODO
        obj->specular_exp = m->roughness;

        sp->material() = obj;

        add_diffuse_texture(sp, m->map_kd, base_filename);
    }
    else if (auto m = std::dynamic_pointer_cast<SubstrateMaterial>(shape->material))
    {
        auto obj = std::make_shared<sg::obj_material>();

        obj->name() = m->name;

        obj->cd = vec3(m->kd.x, m->kd.y, m->kd.z);
        obj->cs = vec3(m->ks.x, m->ks.y, m->ks.z);

        // TODO
        obj->specular_exp = m->uRoughness;

        sp->material() = obj;

        add_diffuse_texture(sp, m->map_kd, base_filename);
    }
    else if (auto m = std::dynamic_pointer_cast<MirrorMaterial>(shape->material))
    {
        auto obj = std::make_shared<sg::obj_material>();

        obj->name() = m->name;

        obj->illum = 3; // indicates purely reflective!

        obj->cs = vec3(m->kr.x, m->kr.y, m->kr.z);

        sp->material() = obj;
    }
    else if (auto m = std::dynamic_pointer_cast<MatteMaterial>(shape->material))
    {
        auto obj = std::make_shared<sg::obj_material>();

        obj->name() = m->name;

        obj->cd = vec3(m->kd.x, m->kd.y, m->kd.z);

        sp->material() = obj;

        add_diffuse_texture(sp, m->map_kd, base_filename);
    }
    else if (auto m = std::dynamic_pointer_cast<GlassMaterial>(shape->material))
    {
        auto glass = std::make_shared<sg::glass_material>();

        glass->name() = m->name;

        glass->ct = vec3(m->kt.x, m->kt.y, m->kt.z);
        glass->cr = vec3(m->kr.x, m->kr.y, m->kr.z);
        glass->ior = vec3(m->index);

        sp->material() = glass;
    }
    else if (auto m = std::dynamic_pointer_cast<UberMaterial>(shape->material))
    {
        auto obj = std::make_shared<sg::obj_material>();

        obj->name() = m->name;

        obj->cd = vec3(m->kd.x, m->kd.y, m->kd.z);
        obj->cs = vec3(m->ks.x, m->ks.y, m->ks.z);

        // TODO
        obj->specular_exp = m->roughness;
        sp->material() = obj;

        add_diffuse_texture(sp, m->map_kd, base_filename);
    }
    else if (auto m = std::dynamic_pointer_cast<MixMaterial>(shape->material))
    {
        // TODO: this just happens to be the case in "landscape"...
        if (auto m0 = std::dynamic_pointer_cast<UberMaterial>(m->material0))
        {
            auto obj = std::make_shared<sg::obj_material>();

            obj->name() = m0->name;

            obj->cd = vec3(m0->kd.x, m0->kd.y, m0->kd.z);
            obj->cs = vec3(m0->ks.x, m0->ks.y, m0->ks.z);

            // TODO
            obj->specular_exp = m0->roughness;
            sp->material() = obj;

            add_diffuse_texture(sp, m0->map_kd, base_filename);
        }
    }
    else
    {
        // Unsupported material or material is null - assign default obj
        sp->material() = std::make_shared<sg::obj_material>();
    }

    return sp;
}

void make_scene_graph(
        Object::SP object,
        sg::node& parent,
        std::unordered_map<Shape::SP, std::shared_ptr<sg::indexed_triangle_mesh>>& shape2itm,
        std::unordered_map<Material::SP, std::shared_ptr<sg::surface_properties>>& mat2prop,
        std::string base_filename
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
                    sp = make_surface_properties(sphere, base_filename);
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
                    itm->normal_indices.resize(mesh->index.size() * 3);
                }

                if (mesh->texcoord.size() > 0)
                {
                    itm->tex_coords = std::make_shared<aligned_vector<vec2>>(mesh->texcoord.size());
                }

                itm->vertex_indices.resize(mesh->index.size() * 3);
                itm->tex_coord_indices.resize(mesh->index.size() * 3); // TODO: in viewer.cpp!

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

                    if (itm->normal_indices.size() > 0)
                    {
                        memcpy(itm->normal_indices.data() + i * 3, &i3.x, sizeof(int) * 3);
                    }

                    memcpy(itm->tex_coord_indices.data() + i * 3, &i3.x, sizeof(int) * 3);
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
                    sp = make_surface_properties(mesh, base_filename);
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

        make_scene_graph(inst->object, *trans, shape2itm, mat2prop, base_filename);

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

        make_scene_graph(scene->world, *root, shape2itm, mat2prop, filename);
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
