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
#include <osg/TriangleFunctor>

#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRViewer.h>

#include <visionaray/detail/aligned_vector.h>
#include <visionaray/math/math.h>
#include <visionaray/bvh.h>
#include <visionaray/kernels.h>
#include <visionaray/point_light.h>
#include <visionaray/render_target.h>
#include <visionaray/scheduler.h>

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
// Functor that stores triangles from osg::Drawable
//

class store_triangle
{
public:

    store_triangle() : triangles_(nullptr) {}

    void init(triangle_list& tris, normal_list& norms, unsigned geom_id)
    {
        triangles_  = &tris;
        normals_    = &norms;
        geom_id_    = geom_id;
    }

    void operator()(osg::Vec3 const& v1, osg::Vec3 const& v2, osg::Vec3 const& v3, bool) const
    {
        assert( triangles_ && normals_ );

        triangle_type tri;
        tri.prim_id = static_cast<unsigned>(triangles_->size());
        tri.geom_id = geom_id_;
        tri.v1 = vec3(v1.x(), v1.y(), v1.z());
        tri.e1 = vec3(v2.x(), v2.y(), v2.z()) - tri.v1;
        tri.e2 = vec3(v3.x(), v3.y(), v3.z()) - tri.v1;
        triangles_->push_back(tri);

        normals_->push_back( normalize(cross(tri.e1, tri.e2)) );

        assert( triangles_->size() == normals_->size() );
    }

private:

    // Store pointers because osg::TriangleFunctor is shitty..
    triangle_list*  triangles_;
    normal_list*    normals_;
    unsigned        geom_id_;

};


//-------------------------------------------------------------------------------------------------
// Visitor to acquire scene data
//

class get_scene_visitor : public osg::NodeVisitor
{
public:

    using base_type = osg::NodeVisitor;
    using base_type::apply;

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
            if (drawable)
            {
                auto set = drawable->getOrCreateStateSet();
                auto attr = set->getAttribute(osg::StateAttribute::MATERIAL);
                auto mat = dynamic_cast<osg::Material*>(attr);

                if (mat)
                {
                    auto cd = mat->getDiffuse(osg::Material::Face::FRONT);
                    auto cs = mat->getSpecular(osg::Material::Face::FRONT);

                    phong<float> vsnray_mat;
                    vsnray_mat.set_cd( vec3(cd.x(), cd.y(), cd.z()) );
                    vsnray_mat.set_kd( 1.0f );
                    vsnray_mat.set_ks( cs.x() ); // TODO: e.g. luminance?
                    vsnray_mat.set_specular_exp( mat->getShininess(osg::Material::Face::FRONT) );
                    materials_.push_back(vsnray_mat);
                }

                assert( static_cast<material_list::size_type>(static_cast<unsigned>(materials.size()) == materials.size()) );

                osg::TriangleFunctor<store_triangle> tf;
                tf.init( triangles_, normals_, std::max(0U, static_cast<unsigned>(materials_.size()) - 1) );
                drawable->accept(tf);
            }
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

struct Visionaray::impl
{
    impl()
    {
    }

    triangle_list                       triangles;
    normal_list                         normals;
    material_list                       materials;
    host_bvh_type                       host_bvh;
    host_sched_type                     host_sched;
    cpu_buffer_rt                       host_rt;

    recti                               viewport;

    osg::ref_ptr<osg::Geode>            geode;
};


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

    ref_ptr<osg::StateSet> state = new osg::StateSet();
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
    // TODO?

    static bool glewed = false;

    if (!glewed)
    {
        glewed = glewInit() == GLEW_OK;
    }


    // Scene data

    if (impl_->triangles.size() == 0)
    {
        // TODO: no dynamic scenes for now :(
        get_scene_visitor visitor(impl_->triangles, impl_->normals, impl_->materials,
            osg::NodeVisitor::TRAVERSE_ALL_CHILDREN);
        visitor.apply(*opencover::cover->getObjectsRoot());

        if (impl_->triangles.size() == 0)
        {
            return;
        }

        if (impl_->materials.size() == 0)
        {
            phong<float> m;
            m.set_cd( vec3(0.8f, 0.8f, 0.8f) );
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
    auto osg_view_matrix = s * t * v;
    auto osg_proj_matrix = opencover::coVRConfig::instance()->channels[0].rightProj;
    auto osg_viewport = osg_cam->getViewport();

    float view[16];
    float proj[16];

    std::copy(osg_view_matrix.ptr(), osg_view_matrix.ptr() + 16, view);
    std::copy(osg_proj_matrix.ptr(), osg_proj_matrix.ptr() + 16, proj);

    mat4 view_matrix(view);
    mat4 proj_matrix(proj);
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

    vec4 lpos;
    glGetLightfv(GL_LIGHT0, GL_POSITION, lpos.data());

    aligned_vector<point_light<float>> lights;
    vec3 light0_pos = (inverse(view_matrix) * lpos).xyz();
    lights.push_back({ light0_pos });


    auto kparams = make_params
    (
        host_primitives.data(),
        host_primitives.data() + host_primitives.size(),
        impl_->normals.data(),
        impl_->materials.data(),
        lights.data(),
        lights.data() + lights.size()
    );

    // Render

    auto kern =  simple::kernel<decltype(kparams)>();
    kern.params = kparams;
    impl_->host_sched.frame(kern, sparams);

    // TODO: generate depth buffer and use RGB render target
    glDepthMask(GL_FALSE);
    glPixelTransferf(GL_ALPHA_SCALE, 0.0f);

    impl_->host_rt.display_color_buffer();
}

}} // namespace visionaray::cover

COVERPLUGIN(visionaray::cover::Visionaray)
