// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <algorithm>
#include <iostream>
#include <limits>
#include <ostream>

#include <boost/algorithm/string.hpp>

#include <GL/glew.h>

#include <osg/io_utils>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/TriangleIndexFunctor>

#include <osgViewer/Renderer>

#include <config/CoviseConfig.h>

#include <cover/coVRConfig.h>
#include <cover/coVRLighting.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRViewer.h>

#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>

#include <visionaray/detail/aligned_vector.h>
#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>
#include <visionaray/bvh.h>
#include <visionaray/kernels.h>
#include <visionaray/point_light.h>
#include <visionaray/render_target.h>
#include <visionaray/scheduler.h>

#include <common/call_kernel.h>
#include <common/render_bvh.h>
#include <common/util.h>

#include "visionaray_plugin.h"

namespace visionaray { namespace cover {


//-------------------------------------------------------------------------------------------------
// Type definitions
//

using triangle_type     = basic_triangle<3, float>;
using triangle_list     = aligned_vector<triangle_type>;
using normal_list       = aligned_vector<vec3>;
using tex_coord_list    = aligned_vector<vec2>;
using material_list     = aligned_vector<phong<float>>;
using texture_list      = aligned_vector<texture<vector<3, unsigned char>, ElementType, 2>>;

using host_ray_type     = basic_ray<simd::float4>;
using host_bvh_type     = bvh<triangle_type>;
using host_sched_type   = tiled_sched<host_ray_type>;


//-------------------------------------------------------------------------------------------------
// Conversion between osg and visionaray
//

vec2 osg_cast(osg::Vec2 const& v)
{
    return vec2( v.x(), v.y() );
}

vec3 osg_cast(osg::Vec3 const& v)
{
    return vec3( v.x(), v.y(), v.z() );
}

vec4 osg_cast(osg::Vec4 const& v)
{
    return vec4( v.x(), v.y(), v.z(), v.w() );
}

mat4 osg_cast(osg::Matrixd const& m)
{
    float arr[16];
    std::copy(m.ptr(), m.ptr() + 16, arr);
    return mat4(arr);
}


//-------------------------------------------------------------------------------------------------
// Functor that stores triangles from osg::Drawable
//

class store_triangle
{
public:

    void init(osg::Vec3Array const* in_vertices, osg::Vec3Array const* in_normals,
        osg::Vec2Array const* in_tex_coords, osg::Matrix const& in_trans_mat, unsigned in_geom_id,
        triangle_list& out_triangles, normal_list& out_normals, tex_coord_list& out_tex_coords)
    {
        in.vertices     = in_vertices;
        in.normals      = in_normals;
        in.tex_coords   = in_tex_coords;
        in.trans_mat    = in_trans_mat;
        in.geom_id      = in_geom_id;
        out.triangles   = &out_triangles;
        out.normals     = &out_normals;
        out.tex_coords  = &out_tex_coords;
    }

    void operator()(unsigned i1, unsigned i2, unsigned i3) const
    {

        // triangles

        assert( in.vertices && out.triangles );
        assert( in.vertices->size() > i1 && in.vertices->size() > i2 && in.vertices->size() > i3 );

        auto v1 = (*in.vertices)[i1] * in.trans_mat;
        auto v2 = (*in.vertices)[i2] * in.trans_mat;
        auto v3 = (*in.vertices)[i3] * in.trans_mat;

        triangle_type tri;
        tri.prim_id = static_cast<unsigned>(out.triangles->size());
        tri.geom_id = in.geom_id;
        tri.v1 = osg_cast(v1);
        tri.e1 = osg_cast(v2) - tri.v1;
        tri.e2 = osg_cast(v3) - tri.v1;
        out.triangles->push_back(tri);


        // normals

        assert( in.normals && out.normals );
        assert( in.normals->size() > i1 && in.normals->size() > i2 && in.normals->size() > i3 );

        auto inv_trans_mat = osg::Matrix::inverse(in.trans_mat);

        // mul left instead of transposing the matrix
        // see http://forum.openscenegraph.org/viewtopic.php?t=2494
        auto n1 = inv_trans_mat * osg::Vec4((*in.normals)[i1], 1.0);
        auto n2 = inv_trans_mat * osg::Vec4((*in.normals)[i2], 1.0);
        auto n3 = inv_trans_mat * osg::Vec4((*in.normals)[i3], 1.0);

        out.normals->push_back( osg_cast(n1).xyz() );
        out.normals->push_back( osg_cast(n2).xyz() );
        out.normals->push_back( osg_cast(n3).xyz() );

        assert( out.triangles->size() == out.normals->size() * 3 );


        // tex coords

        if ( in.tex_coords && in.tex_coords->size() > max(i1, i2, i3) )
        {
            out.tex_coords->push_back( osg_cast((*in.tex_coords)[i1]) );
            out.tex_coords->push_back( osg_cast((*in.tex_coords)[i2]) );
            out.tex_coords->push_back( osg_cast((*in.tex_coords)[i3]) );
        }
        else
        {
            out.tex_coords->push_back( vec2(0.0) );
            out.tex_coords->push_back( vec2(0.0) );
            out.tex_coords->push_back( vec2(0.0) );
        }
    }

private:

    // Store pointers because osg::TriangleIndexFunctor is shitty..

    struct
    {
        osg::Vec3Array const*   vertices    = nullptr;
        osg::Vec3Array const*   normals     = nullptr;
        osg::Vec2Array const*   tex_coords  = nullptr;
        osg::Matrix             trans_mat;
        unsigned                geom_id;
    } in;

    struct
    {
        triangle_list*          triangles   = nullptr;
        normal_list*            normals     = nullptr;
        tex_coord_list*         tex_coords  = nullptr;
    } out;

};


//-------------------------------------------------------------------------------------------------
// Visitor to acquire world transform matrix for a node
//

class get_world_transform_visitor : public osg::NodeVisitor
{
public:

    using base_type = osg::NodeVisitor;
    using base_type::apply;

public:

    get_world_transform_visitor()
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_PARENTS)
    {
    }

    osg::Matrix const& get_matrix() const
    {
        return matrix_;
    }

    void apply(osg::Node& node)
    {
        if (&node == opencover::cover->getObjectsRoot())
        {
            matrix_ = osg::computeLocalToWorld(getNodePath());
        }
        else
        {
            traverse(node);
        }
    }

private:

    osg::Matrix matrix_;

};


//-------------------------------------------------------------------------------------------------
// Visitor to acquire scene data
//

class get_scene_visitor : public osg::NodeVisitor
{
public:

    using base_type = osg::NodeVisitor;
    using base_type::apply;

public:

    get_scene_visitor(triangle_list& triangles, normal_list& normals, tex_coord_list& tex_coords,
        material_list& materials, texture_list& textures, TraversalMode tm)
        : base_type(tm)
        , triangles_(triangles)
        , normals_(normals)
        , tex_coords_(tex_coords)
        , materials_(materials)
        , textures_(textures)
    {
    }

    void apply(osg::Geode& geode)
    {
        for (size_t i = 0; i < geode.getNumDrawables(); ++i)
        {
            auto drawable = geode.getDrawable(i);
            if (!drawable)
            {
                continue;
            }

            auto geom = drawable->asGeometry();
            if (!geom)
            {
                continue;
            }

            auto node_vertices = dynamic_cast<osg::Vec3Array*>(geom->getVertexArray());
            if (!node_vertices || node_vertices->size() == 0)
            {
                continue;
            }

            auto node_normals  = dynamic_cast<osg::Vec3Array*>(geom->getNormalArray());
            if (!node_normals || node_normals->size() == 0)
            {
                continue;
            }

            unsigned tex_unit = 0;
            auto node_tex_coords = dynamic_cast<osg::Vec2Array*>(geom->getTexCoordArray(tex_unit));
            // ok if node_tex_coords == 0

            auto set = geom->getOrCreateStateSet();


            // material

            auto mattr = set->getAttribute(osg::StateAttribute::MATERIAL);
            auto mat = dynamic_cast<osg::Material*>(mattr);

            if (mat)
            {
                auto ca = mat->getAmbient(osg::Material::Face::FRONT);
                auto cd = mat->getDiffuse(osg::Material::Face::FRONT);
                auto cs = mat->getSpecular(osg::Material::Face::FRONT);

                phong<float> vsnray_mat;
                vsnray_mat.set_ca( osg_cast(ca).xyz() );
                vsnray_mat.set_cd( osg_cast(cd).xyz() );
                vsnray_mat.set_ka( 1.0f );
                vsnray_mat.set_kd( M_PI );
                vsnray_mat.set_ks( cs.x() ); // TODO: e.g. luminance?
                vsnray_mat.set_specular_exp( mat->getShininess(osg::Material::Face::FRONT) );
                materials_.push_back(vsnray_mat);
            }


            // texture

            auto tattr = set->getTextureAttribute(0, osg::StateAttribute::TEXTURE);
            auto tex = dynamic_cast<osg::Texture2D*>(tattr);

            if (tex)
            {
                using tex_type = typename texture_list::value_type;

                auto img = tex->getImage();
                tex_type vsnray_tex(img->s(), img->t());
                vsnray_tex.set_address_mode( Clamp );
                vsnray_tex.set_filter_mode( Linear );
                auto data_ptr = reinterpret_cast<tex_type::value_type const*>(img->data());
                vsnray_tex.set_data(data_ptr);
                textures_.push_back(vsnray_tex);
            }
            else
            {
                textures_.push_back( texture<vector<3, unsigned char>, ElementType, 2>(0, 0) );
            }

            assert( materials_.size() == textures_.size() );


            // transform

            get_world_transform_visitor visitor;
            geode.accept(visitor);
            auto world_transform = visitor.get_matrix();


            // geometry

            assert( static_cast<material_list::size_type>(static_cast<unsigned>(materials_.size()) == materials_.size()) );
            unsigned geom_id = materials_.size() == 0 ? 0 : static_cast<unsigned>(materials_.size() - 1);

            osg::TriangleIndexFunctor<store_triangle> tf;
            tf.init( node_vertices, node_normals, node_tex_coords, world_transform, geom_id, triangles_, normals_, tex_coords_ );
            geom->accept(tf);
        }

        base_type::traverse(geode);
    }

private:

    triangle_list&  triangles_;
    normal_list&    normals_;
    tex_coord_list& tex_coords_;
    material_list&  materials_;
    texture_list&   textures_;

};


//-------------------------------------------------------------------------------------------------
// Private implementation
//

struct Visionaray::impl : vrui::coMenuListener
{

    using check_box     = std::unique_ptr<vrui::coCheckboxMenuItem>;
    using menu          = std::unique_ptr<vrui::coMenu>;
    using radio_button  = std::unique_ptr<vrui::coCheckboxMenuItem>;
    using radio_group   = std::unique_ptr<vrui::coCheckboxGroup>;
    using sub_menu      = std::unique_ptr<vrui::coSubMenuItem>;

    enum data_variance { Static, Dynamic };

    impl()
        : host_sched(0)
    {
        init_state_from_config();
        host_sched.set_num_threads(state.num_threads > 0 ? state.num_threads : get_num_processors());
    }

    triangle_list               triangles;
    normal_list                 normals;
    tex_coord_list              tex_coords;
    material_list               materials;
    texture_list                textures;
    host_bvh_type               host_bvh;
    host_sched_type             host_sched;
    cpu_buffer_rt               host_rt;

    mat4                        view_matrix;
    mat4                        proj_matrix;
    recti                       viewport;

    unsigned                    frame_num   = 0;

    osg::ref_ptr<osg::Geode>    geode;

    bool                        glew_init   = false;

    struct
    {
        menu                    main_menu;
        menu                    algo_menu;
        menu                    dev_menu;
        sub_menu                main_menu_entry;
        sub_menu                algo_menu_entry;
        sub_menu                dev_menu_entry;

        // main menu
        check_box               toggle_update_mode;

        // algo menu
        radio_group             algo_group;
        radio_button            simple_button;
        radio_button            whitted_button;
        radio_button            pathtracing_button;

        // dev menu
        check_box               toggle_bvh_display;
        check_box               toggle_normal_display;
        check_box               toggle_tex_coord_display;
    } ui;

    struct
    {
        algorithm               algo            = Simple;
        data_variance           data_var        = Static;
        unsigned                num_threads     = 0;
    } state;

    struct
    {
        bool                    debug_mode      = true;
        bool                    show_bvh        = false;
        bool                    show_normals    = false;
        bool                    show_tex_coords = false;
    } dev_state;

    struct
    {
        GLboolean               depth_mask;
        GLint                   matrix_mode;
        GLboolean               lighting;
    } gl_state;

    void init_state_from_config();
    void init_ui();
    void menuEvent(vrui::coMenuItem* item);

    void store_gl_state();
    void restore_gl_state();
    void update_viewing_params();
    
    template <typename KParams>
    void call_kernel(KParams const& params);

private:

    template <typename KParams>
    void call_kernel_debug(KParams const& params);

};

//-------------------------------------------------------------------------------------------------
// Read state from COVISE config
//

void Visionaray::impl::init_state_from_config()
{

    //
    //
    // <Visionaray>
    //     <DataVariance value="static"  />      <!-- "static" | "dynamic" -->
    //     <Algorithm    value="simple"  />      <!-- "simple" | "whitted" -->
    //     <CPUScheduler numThreads="16" />      <!-- numThreads:Integer   -->
    // </Visioaray>
    //
    //

    // Read config

    using boost::algorithm::to_lower;

    auto algo_str       = covise::coCoviseConfig::getEntry("COVER.Plugin.Visionaray.Algorithm");
    auto data_var_str   = covise::coCoviseConfig::getEntry("COVER.Plugin.Visionaray.DataVariance");
    auto num_threads    = covise::coCoviseConfig::getInt("numThreads", "COVER.Plugin.Visionaray.CPUScheduler", 0);

    to_lower(algo_str);
    to_lower(data_var_str);


    // Update state

    if (algo_str == "whitted")
    {
        state.algo = Whitted;
    }
    else if (algo_str == "pathtracing")
    {
        state.algo = Pathtracing;
    }
    else
    {
        state.algo = Simple;
    }

    state.data_var      = data_var_str == "dynamic" ? Dynamic : Static;
    state.num_threads   = num_threads;
}

void Visionaray::impl::init_ui()
{
    using namespace vrui;

    ui.main_menu_entry.reset(new coSubMenuItem("Visionaray..."));
    opencover::cover->getMenu()->add(ui.main_menu_entry.get());

    // main menu

    ui.main_menu.reset(new coRowMenu("Visionaray", opencover::cover->getMenu()));
    ui.main_menu_entry->setMenu(ui.main_menu.get());


    ui.toggle_update_mode.reset(new coCheckboxMenuItem("Update scene per frame", state.data_var == impl::Dynamic));
    ui.toggle_update_mode->setMenuListener(this);
    ui.main_menu->add(ui.toggle_update_mode.get());


    // algorithm submenu

    ui.algo_menu_entry.reset(new coSubMenuItem("Rendering algorithm..."));
    ui.main_menu->add(ui.algo_menu_entry.get());

    ui.algo_menu.reset(new coRowMenu("Rendering algorithm", ui.main_menu.get()));
    ui.algo_menu_entry->setMenu(ui.algo_menu.get());


    ui.algo_group.reset(new coCheckboxGroup( /* allow empty selection: */ false ));

    ui.simple_button.reset(new coCheckboxMenuItem("Simple", state.algo == Simple, ui.algo_group.get()));
    ui.simple_button->setMenuListener(this);
    ui.algo_menu->add(ui.simple_button.get());

    ui.whitted_button.reset(new coCheckboxMenuItem("Whitted", state.algo == Whitted, ui.algo_group.get()));
    ui.whitted_button->setMenuListener(this);
    ui.algo_menu->add(ui.whitted_button.get());

    ui.pathtracing_button.reset(new coCheckboxMenuItem("Pathtracing", state.algo == Pathtracing, ui.algo_group.get()));
    ui.pathtracing_button->setMenuListener(this);
    ui.algo_menu->add(ui.pathtracing_button.get());


    // dev submenu at the bottom!

    if (dev_state.debug_mode)
    {
        ui.dev_menu_entry.reset(new coSubMenuItem("Developer..."));
        ui.main_menu->add(ui.dev_menu_entry.get());

        ui.dev_menu.reset(new coRowMenu("Developer", ui.main_menu.get()));
        ui.dev_menu_entry->setMenu(ui.dev_menu.get());


        ui.toggle_bvh_display.reset(new coCheckboxMenuItem("Show BVH outlines", false));
        ui.toggle_bvh_display->setMenuListener(this);
        ui.dev_menu->add(ui.toggle_bvh_display.get());

        ui.toggle_normal_display.reset(new coCheckboxMenuItem("Show surface normals", false));
        ui.toggle_normal_display->setMenuListener(this);
        ui.dev_menu->add(ui.toggle_normal_display.get());

        ui.toggle_tex_coord_display.reset(new coCheckboxMenuItem("Show texture coordinates", false));
        ui.toggle_tex_coord_display->setMenuListener(this);
        ui.dev_menu->add(ui.toggle_tex_coord_display.get());
    }
}

void Visionaray::impl::store_gl_state()
{
    glGetBooleanv(GL_DEPTH_WRITEMASK, &gl_state.depth_mask);
    glGetIntegerv(GL_MATRIX_MODE, &gl_state.matrix_mode);
    gl_state.lighting = glIsEnabled(GL_LIGHTING);
}

void Visionaray::impl::restore_gl_state()
{
    if (gl_state.lighting)
    {
        glEnable(GL_LIGHTING);
    }
    else
    {
        glDisable(GL_LIGHTING);
    }

    glMatrixMode(gl_state.matrix_mode);
    glDepthMask(gl_state.depth_mask);
}

void Visionaray::impl::update_viewing_params()
{
    auto osg_cam = opencover::coVRConfig::instance()->channels[0].camera;

    // Matrices

    auto t = opencover::cover->getXformMat();
    auto s = opencover::cover->getObjectsScale()->getMatrix();
    // TODO: understand COVER API..
//  auto v = opencover::cover->getViewerMat();
    auto v = opencover::coVRConfig::instance()->channels[0].rightView;
    auto view = osg_cast( s * t * v );
    auto proj = osg_cast( opencover::coVRConfig::instance()->channels[0].rightProj );

    if (state.data_var == impl::Static && view_matrix == view && proj_matrix == proj)
    {
        ++frame_num;
    }
    else
    {
        frame_num = 0;
    }

    view_matrix = view;
    proj_matrix = proj;


    // Viewport

    auto osg_viewport = osg_cam->getViewport();
    recti vp(osg_viewport->x(), osg_viewport->y(), osg_viewport->width(), osg_viewport->height());

    if (viewport != vp)
    {
        viewport = vp;
        host_rt.resize(viewport[2], viewport[3]);
    }
}

//-------------------------------------------------------------------------------------------------
// Call either one of the visionaray kernels or a custom one
//

template <typename KParams>
void Visionaray::impl::call_kernel(KParams const& params)
{
    if (dev_state.debug_mode && (dev_state.show_normals || dev_state.show_tex_coords))
    {
        call_kernel_debug( params );
    }
    else
    {
        visionaray::call_kernel( state.algo, host_sched, params, frame_num, view_matrix, proj_matrix, viewport, host_rt );
    }
}

//-------------------------------------------------------------------------------------------------
// Custom kernels to debug some internal values
//

template <typename KParams>
void Visionaray::impl::call_kernel_debug(KParams const& params)
{
    using R = host_ray_type;
    using S = typename R::scalar_type;
    using C = vector<4, S>;

    auto sparams = make_sched_params<pixel_sampler::uniform_type>( view_matrix, proj_matrix, viewport, host_rt );

    if (dev_state.show_normals)
    {
        host_sched.frame([&](R ray) -> C
        {
            auto hit_rec = closest_hit(ray, params.prims.begin, params.prims.end);
            auto surf = get_surface(hit_rec, params);
            return select( hit_rec.hit, C(surf.normal, S(1.0)), C(0.0) );
        },
        sparams);
    }
    else if (dev_state.show_tex_coords)
    {
        host_sched.frame([&](R ray) -> C
        {
            auto hit_rec = closest_hit(ray, params.prims.begin, params.prims.end);
            auto tc = get_tex_coord(params.tex_coords, hit_rec);
            return select( hit_rec.hit, C(tc, S(1.0), S(1.0)), C(0.0) );
        },
        sparams);
    }
}

void Visionaray::impl::menuEvent(vrui::coMenuItem* item)
{
    // main menu
    if (item == ui.toggle_update_mode.get())
    {
        state.data_var = ui.toggle_update_mode->getState() ? impl::Dynamic : impl::Static;
    }

    // algorithm submenu
    if (item == ui.simple_button.get())
    {
        state.algo = Simple;
    }
    else if (item == ui.whitted_button.get())
    {
        state.algo = Whitted;
    }
    else if (item == ui.pathtracing_button.get())
    {
        state.algo = Pathtracing;
    }

    // dev submenu
    if (item == ui.toggle_bvh_display.get())
    {
        dev_state.show_bvh = ui.toggle_bvh_display->getState();
    }

    if (item == ui.toggle_normal_display.get())
    {
        dev_state.show_normals = ui.toggle_normal_display->getState();
    }

    if (item == ui.toggle_tex_coord_display.get())
    {
        dev_state.show_tex_coords = ui.toggle_tex_coord_display->getState();
    }
}


//-------------------------------------------------------------------------------------------------
// Visionaray plugin
//

Visionaray::Visionaray()
    : impl_(new impl)
{
    setSupportsDisplayList(false);
}

Visionaray::Visionaray(Visionaray const& rhs, osg::CopyOp const& op)
    : Drawable(rhs, op)
    , impl_(new impl)
{
    setSupportsDisplayList(false);
}

Visionaray::~Visionaray()
{
    impl_->geode->removeDrawable(this);
    opencover::cover->getObjectsRoot()->removeChild(impl_->geode);
}

bool Visionaray::init()
{
    using namespace osg;

    opencover::VRViewer::instance()->culling(false);

    std::cout << "Init Visionaray Plugin!!" << std::endl;

    impl_->init_ui();

    impl_->geode = new osg::Geode;
    impl_->geode->setName("Visionaray");
    impl_->geode->addDrawable(this);

    opencover::cover->getScene()->addChild(impl_->geode);

    return true;
}

void Visionaray::preFrame()
{
}

void Visionaray::expandBoundingSphere(osg::BoundingSphere &bs)
{
    aabb bounds( vec3(std::numeric_limits<float>::max()), -vec3(std::numeric_limits<float>::max()) );
    for (auto const& tri : impl_->triangles)
    {
        auto v1 = tri.v1;
        auto v2 = tri.v1 + tri.e1;
        auto v3 = tri.v1 + tri.e2;

        bounds = combine(bounds, aabb(v1, v1));
        bounds = combine(bounds, aabb(v2, v2));
        bounds = combine(bounds, aabb(v3, v3));
    }

    auto c = bounds.center();
    osg::BoundingSphere::vec_type center(c.x, c.y, c.z);
    osg::BoundingSphere::value_type radius = length( c - bounds.min );
    bs.set(center, radius);
}

void Visionaray::drawImplementation(osg::RenderInfo&) const
{
    if (!impl_->glew_init)
    {
        impl_->glew_init = glewInit() == GLEW_OK;
    }

    if (!impl_->glew_init)
    {
        return;
    }

    impl_->store_gl_state();


    // Scene data

    if (impl_->state.data_var == impl::Dynamic || impl_->triangles.size() == 0)
    {
        // TODO: real dynamic scenes :)

        impl_->triangles.clear();
        impl_->normals.clear();
        impl_->materials.clear();

        get_scene_visitor visitor(impl_->triangles, impl_->normals, impl_->tex_coords, impl_->materials,
            impl_->textures, osg::NodeVisitor::TRAVERSE_ALL_CHILDREN);
        opencover::cover->getObjectsRoot()->accept(visitor);

        if (impl_->triangles.size() == 0)
        {
            return;
        }

        if (impl_->materials.size() == 0)
        {
            phong<float> m;
            m.set_ca( vec3(0.2f, 0.2f, 0.2f) );
            m.set_cd( vec3(0.8f, 0.8f, 0.8f) );
            m.set_ka( 1.0f );
            m.set_kd( M_PI );
            m.set_ks( 1.0f );
            m.set_specular_exp( 32.0f );
            impl_->materials.push_back(m);
        }

        impl_->host_bvh = build<host_bvh_type>(impl_->triangles.data(), impl_->triangles.size());

        opencover::cover->getObjectsRoot()->setNodeMask
        (
            opencover::cover->getObjectsRoot()->getNodeMask()
         & ~opencover::VRViewer::instance()->getCullMask()
        );
    }


    // Camera matrices, viewport, render target resize

    impl_->update_viewing_params();


    // Kernel params

    aligned_vector<host_bvh_type::bvh_ref> host_primitives;
    host_primitives.push_back(impl_->host_bvh.ref());

    aligned_vector<point_light<float>> lights;

    auto add_light = [&](vec4 lpos, vec4 ldiff)
    {
        // map OpenGL [-1,1] to Visionaray [0,1]
        ldiff += 1.0f;
        ldiff /= 2.0f;

        point_light<float> light;
        light.set_position( lpos.xyz() );
        light.set_cl( ldiff.xyz() );
        light.set_kl( ldiff.w );

        lights.push_back(light);
    };

    if (opencover::coVRLighting::instance()->headlightState)
    {
        auto renderer = dynamic_cast<osgViewer::Renderer*>(opencover::coVRConfig::instance()->channels[0].camera->getRenderer());
        assert(renderer);

        auto scene_view = renderer->getSceneView(0);
        auto headlight = scene_view->getLight();

        auto hlpos  = inverse(impl_->view_matrix) * vec4(osg_cast(headlight->getPosition()).xyz(), 1.0f);
        auto hldiff = osg_cast(headlight->getDiffuse());

        add_light(hlpos, hldiff);
    }

    auto cover_lights = opencover::coVRLighting::instance()->lightList;
    for (auto it = cover_lights.begin(); it != cover_lights.end(); ++it)
    {
        if ((*it).source == opencover::coVRLighting::instance()->headlight)
        {
            continue;
        }

        if ((*it).on)
        {
            auto l = (*it).source->getLight();

            // TODO
            auto lpos  = osg_cast(l->getPosition());
            auto ldiff = osg_cast(l->getDiffuse());

            add_light(lpos, ldiff);
        }
    }

    auto kparams = make_params<normals_per_vertex_binding>
    (
        host_primitives.data(),
        host_primitives.data() + host_primitives.size(),
        impl_->normals.data(),
        impl_->tex_coords.data(),
        impl_->materials.data(),
        impl_->textures.data(),
        lights.data(),
        lights.data() + lights.size(),
        vec4(0.0f)
    );


    // Render
    impl_->call_kernel(kparams);

    // TODO: generate depth buffer
    glDepthMask(GL_FALSE);

    impl_->host_rt.display_color_buffer();

    glDepthMask(GL_TRUE);

    if (impl_->dev_state.debug_mode && impl_->dev_state.show_bvh)
    {
        glDisable(GL_LIGHTING);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadMatrixf(impl_->proj_matrix.data());

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadMatrixf(impl_->view_matrix.data());

        glColor3f(1.0f, 1.0f, 1.0f);
        render_bvh(impl_->host_bvh);

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
    }

    impl_->restore_gl_state();
}

}} // namespace visionaray::cover

COVERPLUGIN(visionaray::cover::Visionaray)
