// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <algorithm>
#include <iostream>
#include <limits>
#include <ostream>

#include <GL/glew.h>

#include <osg/io_utils>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/StateSet>
#include <osg/TriangleIndexFunctor>

#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRViewer.h>

#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>

#include <visionaray/detail/aligned_vector.h>
#include <visionaray/math/math.h>
#include <visionaray/bvh.h>
#include <visionaray/kernels.h>
#include <visionaray/point_light.h>
#include <visionaray/render_target.h>
#include <visionaray/scheduler.h>

#include <common/render_bvh.h>

#include "visionaray_plugin.h"

namespace visionaray { namespace cover {


//-------------------------------------------------------------------------------------------------
// Type definitions
//

using triangle_type     = basic_triangle<3, float>;
using triangle_list     = aligned_vector<triangle_type>;
using normal_list       = aligned_vector<vec3>;
using material_list     = aligned_vector<phong<float>>;

using host_ray_type     = basic_ray<simd::float4>;
using host_bvh_type     = bvh<triangle_type>;
using host_sched_type   = tiled_sched<host_ray_type>;


//-------------------------------------------------------------------------------------------------
// Conversion between osg and visionaray
//

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

    store_triangle()
        : in({ nullptr, nullptr, osg::Matrix(), 0 })
        , out({ nullptr, nullptr })
    {}

    void init(osg::Vec3Array* in_vertices, osg::Vec3Array* in_normals,
        osg::Matrix const& in_trans_mat, unsigned in_geom_id,
        triangle_list& out_triangles, normal_list& out_normals)
    {
        in.vertices     = in_vertices;
        in.normals      = in_normals;
        in.trans_mat    = in_trans_mat;
        in.geom_id      = in_geom_id;
        out.triangles   = &out_triangles;
        out.normals     = &out_normals;
    }

    void operator()(unsigned i1, unsigned i2, unsigned i3) const
    {
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
    }

private:

    // Store pointers because osg::TriangleIndexFunctor is shitty..

    struct
    {
        osg::Vec3Array* vertices;
        osg::Vec3Array* normals;
        osg::Matrix     trans_mat;
        unsigned        geom_id;
    } in;

    struct
    {
        triangle_list*  triangles;
        normal_list*    normals;
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

    get_scene_visitor(triangle_list& tris, normal_list& norms, material_list& mats, TraversalMode tm)
        : base_type(tm)
        , triangles_(tris)
        , normals_(norms)
        , materials_(mats)
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

            auto set = geom->getOrCreateStateSet();
            auto attr = set->getAttribute(osg::StateAttribute::MATERIAL);
            auto mat = dynamic_cast<osg::Material*>(attr);

            if (mat)
            {
                auto ca = mat->getAmbient(osg::Material::Face::FRONT);
                auto cd = mat->getDiffuse(osg::Material::Face::FRONT);
                auto cs = mat->getSpecular(osg::Material::Face::FRONT);

                phong<float> vsnray_mat;
                vsnray_mat.set_ca( osg_cast(ca).xyz() );
                vsnray_mat.set_cd( osg_cast(cd).xyz() );
                vsnray_mat.set_ka( 1.0f );
                vsnray_mat.set_kd( 1.0f );
                vsnray_mat.set_ks( cs.x() ); // TODO: e.g. luminance?
                vsnray_mat.set_specular_exp( mat->getShininess(osg::Material::Face::FRONT) );
                materials_.push_back(vsnray_mat);
            }

            get_world_transform_visitor visitor;
            geode.accept(visitor);
            auto world_transform = visitor.get_matrix();

            assert( static_cast<material_list::size_type>(static_cast<unsigned>(materials_.size()) == materials_.size()) );
            unsigned geom_id = materials_.size() == 0 ? 0 : static_cast<unsigned>(materials_.size() - 1);

            osg::TriangleIndexFunctor<store_triangle> tf;
            tf.init( node_vertices, node_normals, world_transform, geom_id, triangles_, normals_ );
            geom->accept(tf);
        }

        base_type::traverse(geode);
    }

private:

    triangle_list&  triangles_;
    normal_list&    normals_;
    material_list&  materials_;

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

    enum algorithm { Simple, Whitted };

    impl()
        : state({ Simple, false })
        , dev_state({ true, false, false })
    {
    }

    triangle_list               triangles;
    normal_list                 normals;
    material_list               materials;
    host_bvh_type               host_bvh;
    host_sched_type             host_sched;
    cpu_buffer_rt               host_rt;

    recti                       viewport;

    osg::ref_ptr<osg::Geode>    geode;

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

        // dev menu
        check_box               toggle_bvh_display;
        check_box               toggle_normal_display;
    } ui;

    struct
    {
        algorithm               algo;
        bool                    update_scene_per_frame;
    } state;

    struct
    {
        bool                    debug_mode;
        bool                    show_bvh;
        bool                    show_normals;
    } dev_state;

    struct
    {
        GLboolean               depth_mask;
        GLint                   matrix_mode;
        GLboolean               lighting;
    } gl_state;

    void init_ui();
    void menuEvent(vrui::coMenuItem* item);

    void store_gl_state();
    void restore_gl_state();

};

void Visionaray::impl::init_ui()
{
    using namespace vrui;

    ui.main_menu_entry.reset(new coSubMenuItem("Visionaray..."));
    opencover::cover->getMenu()->add(ui.main_menu_entry.get());

    // main menu

    ui.main_menu.reset(new coRowMenu("Visionaray", opencover::cover->getMenu()));
    ui.main_menu_entry->setMenu(ui.main_menu.get());


    ui.toggle_update_mode.reset(new coCheckboxMenuItem("Update scene per frame", false));
    ui.toggle_update_mode->setMenuListener(this);
    ui.main_menu->add(ui.toggle_update_mode.get());


    // algorithm submenu

    ui.algo_menu_entry.reset(new coSubMenuItem("Rendering algorithm..."));
    ui.main_menu->add(ui.algo_menu_entry.get());

    ui.algo_menu.reset(new coRowMenu("Rendering algorithm", ui.main_menu.get()));
    ui.algo_menu_entry->setMenu(ui.algo_menu.get());


    ui.algo_group.reset(new coCheckboxGroup( /* allow empty selection: */ false ));

    ui.simple_button.reset(new coCheckboxMenuItem("Simple", false, ui.algo_group.get()));
    ui.simple_button->setMenuListener(this);
    ui.algo_menu->add(ui.simple_button.get());

    ui.whitted_button.reset(new coCheckboxMenuItem("Whitted", false, ui.algo_group.get()));
    ui.whitted_button->setMenuListener(this);
    ui.algo_menu->add(ui.whitted_button.get());


    // dev submenu at the bottom!

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

void Visionaray::impl::menuEvent(vrui::coMenuItem* item)
{
    // main menu
    if (item == ui.toggle_update_mode.get())
    {
        state.update_scene_per_frame = ui.toggle_update_mode->getState();
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

    // dev submenu
    if (item == ui.toggle_bvh_display.get())
    {
        dev_state.show_bvh = ui.toggle_bvh_display->getState();
    }

    if (item == ui.toggle_normal_display.get())
    {
        dev_state.show_normals = ui.toggle_normal_display->getState();
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
    opencover::cover->getObjectsRoot()->removeChild(impl_->geode);
}

bool Visionaray::init()
{
    using namespace osg;

    opencover::VRViewer::instance()->culling(false);

    std::cout << "Init Visionaray Plugin!!" << std::endl;

    impl_->init_ui();

    ref_ptr<osg::StateSet> state = new osg::StateSet;
    state->setGlobalDefaults();

    impl_->geode = new osg::Geode;
    impl_->geode->setName("Visionaray");
    impl_->geode->setStateSet(state);
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
    impl_->store_gl_state();

    // TODO?

    static bool glewed = false;

    if (!glewed)
    {
        glewed = glewInit() == GLEW_OK;
    }


    // Scene data

    if (impl_->state.update_scene_per_frame || impl_->triangles.size() == 0)
    {
        // TODO: real dynamic scenes :)

        impl_->triangles.clear();
        impl_->normals.clear();
        impl_->materials.clear();

        get_scene_visitor visitor(impl_->triangles, impl_->normals, impl_->materials,
            osg::NodeVisitor::TRAVERSE_ALL_CHILDREN);
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
            m.set_kd( 1.0f );
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


    // Sched params

    auto osg_cam = opencover::coVRConfig::instance()->channels[0].camera;

    auto t = opencover::cover->getXformMat();
    auto s = opencover::cover->getObjectsScale()->getMatrix();
    // TODO: understand COVER API..
//  auto v = opencover::cover->getViewerMat();
    auto v = opencover::coVRConfig::instance()->channels[0].rightView;
    auto view_matrix = osg_cast( s * t * v );
    auto proj_matrix = osg_cast( opencover::coVRConfig::instance()->channels[0].rightProj );

    auto osg_viewport = osg_cam->getViewport();
    recti viewport(osg_viewport->x(), osg_viewport->y(), osg_viewport->width(), osg_viewport->height());

    auto sparams = make_sched_params<pixel_sampler::uniform_type>( view_matrix, proj_matrix, viewport, impl_->host_rt );

    if (impl_->viewport != viewport)
    {
        impl_->host_rt.resize(viewport[2], viewport[3]);
        impl_->viewport = viewport;
    }

    // Kernel params

    aligned_vector<host_bvh_type::bvh_ref> host_primitives;
    host_primitives.push_back(impl_->host_bvh.ref());

    aligned_vector<point_light<float>> lights;

    int max_lights = 8;
    glGetIntegerv(GL_MAX_LIGHTS, &max_lights);

    for (int i = 0; i < max_lights; ++i)
    {
        if (glIsEnabled(GL_LIGHT0 + i))
        {
            vec4 lpos;
            glGetLightfv(GL_LIGHT0 + i, GL_POSITION, lpos.data());

            vec4 ldiff;
            glGetLightfv(GL_LIGHT0 + i, GL_DIFFUSE, ldiff.data());

            // map OpenGL [-1,1] to Visionaray [0,1]
            ldiff += 1.0f;
            ldiff /= 2.0f;

            point_light<float> light;
            light.set_position( lpos.xyz() );
            light.set_cl( ldiff.xyz() );
            light.set_kl( ldiff.w );

            lights.push_back(light);
        }
    }

    auto kparams = make_params<normals_per_vertex_binding>
    (
        host_primitives.data(),
        host_primitives.data() + host_primitives.size(),
        impl_->normals.data(),
        impl_->materials.data(),
        lights.data(),
        lights.data() + lights.size(),
        vec4(0.0f)
    );

    // Render

    if (impl_->dev_state.debug_mode && impl_->dev_state.show_normals)
    {
        using R = host_ray_type;
        using S = typename R::scalar_type;
        using C = vector<4, S>;
        impl_->host_sched.frame([&](R ray) -> C
        {
            auto hit_rec = closest_hit(ray, kparams.prims.begin, kparams.prims.end);
            auto surf = get_surface(hit_rec, kparams);
            return select( hit_rec.hit, C(surf.normal, S(1.0)), C(0.0) );
        },
        sparams);
    }
    else if (impl_->state.algo == impl::Simple)
    {
        auto kern = simple::kernel<decltype(kparams)>();
        kern.params = kparams;
        impl_->host_sched.frame(kern, sparams);
    }
    else if (impl_->state.algo == impl::Whitted)
    {
        auto kern =  whitted::kernel<decltype(kparams)>();
        kern.params = kparams;
        impl_->host_sched.frame(kern, sparams);
    }
    else
    {
        // TODO: inform user
    }

    // TODO: generate depth buffer
    glDepthMask(GL_FALSE);

    impl_->host_rt.display_color_buffer();

    glDepthMask(GL_TRUE);

    if (impl_->dev_state.debug_mode && impl_->dev_state.show_bvh)
    {
        glDisable(GL_LIGHTING);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadMatrixf(proj_matrix.data());

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadMatrixf(view_matrix.data());

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
