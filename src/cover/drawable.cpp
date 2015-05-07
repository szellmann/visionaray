// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <algorithm>
#include <limits>
#include <memory>

#include <GL/glew.h>

#include <osg/io_utils>
#include <osg/LightModel>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Switch>
#include <osg/TriangleIndexFunctor>

#include <osgViewer/Renderer>

#include <cover/coVRConfig.h>
#include <cover/coVRLighting.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRViewer.h>

#include <visionaray/detail/aligned_vector.h>
#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>
#include <visionaray/bvh.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/kernels.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>

#include <common/call_kernel.h>
#include <common/render_bvh.h>
#include <common/util.h>

#include "drawable.h"

namespace visionaray { namespace cover {


//-------------------------------------------------------------------------------------------------
// Type definitions
//

using triangle_type             = basic_triangle<3, float>;
using triangle_list             = aligned_vector<triangle_type>;
using normal_list               = aligned_vector<vec3>;
using tex_coord_list            = aligned_vector<vec2>;
using material_list             = aligned_vector<plastic<float>>;
using texture_list              = aligned_vector<texture<vector<3, unorm<8>>, ElementType, 2>>;

using host_ray_type             = basic_ray<simd::float4>;
using host_bvh_type             = index_bvh<triangle_type>;
using host_render_target_type   = cpu_buffer_rt<PF_RGBA32F, PF_DEPTH32F>;
using host_sched_type           = tiled_sched<host_ray_type>;


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

tex_address_mode osg_cast(osg::Texture::WrapMode mode)
{
    switch (mode)
    {

    default:
        // fall-through
    case osg::Texture::CLAMP:
        return visionaray::Clamp;

    case osg::Texture::REPEAT:
        return visionaray::Wrap;

    case osg::Texture::MIRROR:
        return visionaray::Mirror;

    }
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
// Visitor to check visibility by traversing upwards to a node's parents
//

class visibility_visitor : public osg::NodeVisitor
{
public:

    using base_type = osg::NodeVisitor;
    using base_type::apply;

public:

    visibility_visitor(osg::Node* node)
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_PARENTS)
        , last_child_(node)
        , visible_(true)
    {
    }

    bool is_visible() const
    {
        return visible_;
    }

    void apply(osg::Node& node)
    {
        auto sw = dynamic_cast<osg::Switch*>(&node);
        if (sw && sw->containsNode(last_child_))
        {
            visible_ &= sw->getChildValue(last_child_);
        }

        if (visible_)
        {
            last_child_ = &node;
            traverse(node);
        }
    }

private:

    osg::Node* last_child_;
    bool visible_;

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


            // Simple checks are done - traverse parents to see if node is visible

            visibility_visitor visible(&geode);
            geode.accept(visible);

            if (!visible.is_visible())
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

                plastic<float> vsnray_mat;
                vsnray_mat.set_ca( from_rgb(osg_cast(ca).xyz()) );
                vsnray_mat.set_cd( from_rgb(osg_cast(cd).xyz()) );
                vsnray_mat.set_cs( from_rgb(osg_cast(cs).xyz()) );
                vsnray_mat.set_ka( 1.0f );
                vsnray_mat.set_kd( 1.0f );
                vsnray_mat.set_ks( 1.0f );
                vsnray_mat.set_specular_exp( mat->getShininess(osg::Material::Face::FRONT) );
                materials_.push_back(vsnray_mat);
            }


            // texture

            auto tattr = set->getTextureAttribute(0, osg::StateAttribute::TEXTURE);
            auto tex = dynamic_cast<osg::Texture2D*>(tattr);
            auto img = tex != nullptr ? tex->getImage() : nullptr;

            if (tex && img && img->getPixelFormat() == GL_RGB)
            {
                using tex_type = typename texture_list::value_type;

                tex_type vsnray_tex(img->s(), img->t());
                vsnray_tex.set_address_mode( 0, osg_cast(tex->getWrap(osg::Texture::WRAP_S)) );
                vsnray_tex.set_address_mode( 1, osg_cast(tex->getWrap(osg::Texture::WRAP_T)) );
                vsnray_tex.set_filter_mode( Linear );
                auto data_ptr = reinterpret_cast<tex_type::value_type const*>(img->data());
                vsnray_tex.set_data(data_ptr);
                textures_.push_back(vsnray_tex);
            }
            else
            {
                textures_.push_back( texture<vector<3, unorm<8>>, ElementType, 2>(0, 0) );
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

struct drawable::impl
{

    impl()
        : host_sched(0)
    {
    }

    triangle_list                   triangles;
    normal_list                     normals;
    tex_coord_list                  tex_coords;
    material_list                   materials;
    texture_list                    textures;
    host_bvh_type                   host_bvh;
    host_sched_type                 host_sched;
    host_render_target_type         host_rt;

    mat4                            view_matrix;
    mat4                            proj_matrix;
    recti                           viewport;

    unsigned                        frame_num       = 0;

    algorithm                       algo_current    = Simple;

    bvh_outline_renderer            outlines;

    bool                            glew_init       = false;

    std::shared_ptr<render_state>   state           = nullptr;
    std::shared_ptr<debug_state>    dev_state       = nullptr;
    struct
    {
        GLint                       matrix_mode;
        GLboolean                   lighting;
    } gl_state;

    void update_state(
            std::shared_ptr<render_state> const& state,
            std::shared_ptr<debug_state>  const& dev_state
            )
    {
        this->state     = state;
        this->dev_state = dev_state;

        if (state)
        {
            host_sched.set_num_threads(
                    state->num_threads > 0
                  ? state->num_threads
                  : get_num_processors()
                    );
        }
    }

    void store_gl_state();
    void restore_gl_state();
    void update_viewing_params();
    
    template <typename KParams>
    void call_kernel(KParams const& params);

private:

    template <typename KParams>
    void call_kernel_debug(KParams const& params);

};


void drawable::impl::store_gl_state()
{
    glGetIntegerv(GL_MATRIX_MODE, &gl_state.matrix_mode);
    gl_state.lighting = glIsEnabled(GL_LIGHTING);
}

void drawable::impl::restore_gl_state()
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
}

void drawable::impl::update_viewing_params()
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


    // Viewport

    auto osg_viewport = osg_cam->getViewport();
    recti vp(osg_viewport->x(), osg_viewport->y(), osg_viewport->width(), osg_viewport->height());


    // Reset frame counter on change or if scene is dynamic

    if (state->data_var == Dynamic || state->algo != algo_current
     || view_matrix != view || proj_matrix != proj || viewport != vp)
    {
        frame_num = 0;
    }


    // Update

    algo_current = state->algo;
    view_matrix  = view;
    proj_matrix  = proj;

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
void drawable::impl::call_kernel(KParams const& params)
{
    if (dev_state->debug_mode && (dev_state->show_normals || dev_state->show_tex_coords))
    {
        call_kernel_debug( params );
    }
    else
    {
        visionaray::call_kernel( state->algo, host_sched, params, frame_num, view_matrix, proj_matrix, viewport, host_rt );
    }
}


//-------------------------------------------------------------------------------------------------
// Custom kernels to debug some internal values
//

template <typename KParams>
void drawable::impl::call_kernel_debug(KParams const& params)
{
    using R = host_ray_type;
    using S = typename R::scalar_type;
    using C = vector<4, S>;

    auto sparams = make_sched_params<pixel_sampler::uniform_type>( view_matrix, proj_matrix, viewport, host_rt );

    if (dev_state->show_normals)
    {
        host_sched.frame([&](R ray) -> result_record<S>
        {
            result_record<S> result;
            auto hit_rec        = closest_hit(ray, params.prims.begin, params.prims.end);
            auto surf           = get_surface(hit_rec, params);
            result.hit          = hit_rec.hit;
            result.color        = select( hit_rec.hit, C(surf.normal, S(1.0)), C(0.0) );
            result.isect_pos    = hit_rec.isect_pos;
            return result;
        },
        sparams);
    }
    else if (dev_state->show_tex_coords)
    {
        host_sched.frame([&](R ray) -> result_record<S>
        {
            result_record<S> result;
            auto hit_rec        = closest_hit(ray, params.prims.begin, params.prims.end);
            auto tc             = get_tex_coord(params.tex_coords, hit_rec);
            result.hit          = hit_rec.hit;
            result.color        = select( hit_rec.hit, C(tc, S(1.0), S(1.0)), C(0.0) );
            result.isect_pos    = hit_rec.isect_pos;
            return result;
        },
        sparams);
    }
}


//-------------------------------------------------------------------------------------------------
//
//

drawable::drawable()
    : impl_(new impl)
{
    setSupportsDisplayList(false);
}

drawable::~drawable()
{
}

void drawable::update_state(
        std::shared_ptr<render_state> const& state,
        std::shared_ptr<debug_state>  const& dev_state
        )
{
    impl_->update_state(state, dev_state);
}

void drawable::expandBoundingSphere(osg::BoundingSphere &bs)
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


//-------------------------------------------------------------------------------------------------
// Private osg::Drawable interface
//

drawable* drawable::cloneType() const
{
    return new drawable;
}

osg::Object* drawable::clone(const osg::CopyOp& op) const
{
    return new drawable(*this, op);
}

drawable::drawable(drawable const& rhs, osg::CopyOp const& op)
    : osg::Drawable(rhs, op)
{
    setSupportsDisplayList(false);
}


//-------------------------------------------------------------------------------------------------
// Draw implementation
//

void drawable::drawImplementation(osg::RenderInfo&) const
{
    if (!impl_->state || !impl_->dev_state)
    {
        return;
    }

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

    if (impl_->state->data_var == Dynamic || impl_->triangles.size() == 0)
    {
        // TODO: real dynamic scenes :)

        impl_->triangles.clear();
        impl_->normals.clear();
        impl_->tex_coords.clear();
        impl_->materials.clear();
        impl_->textures.clear();

        get_scene_visitor visitor(
                impl_->triangles,
                impl_->normals,
                impl_->tex_coords,
                impl_->materials,
                impl_->textures,
                osg::NodeVisitor::TRAVERSE_ALL_CHILDREN
                );
        opencover::cover->getObjectsRoot()->accept(visitor);

        if (impl_->triangles.size() == 0)
        {
            return;
        }

        if (impl_->materials.size() == 0)
        {
            plastic<float> m;
            m.set_ca( from_rgb(0.2f, 0.2f, 0.2f) );
            m.set_cd( from_rgb(0.8f, 0.8f, 0.8f) );
            m.set_cs( from_rgb(0.1f, 0.1f, 0.1f) );
            m.set_ka( 1.0f );
            m.set_kd( 1.0f );
            m.set_ks( 1.0f );
            m.set_specular_exp( 32.0f );
            impl_->materials.push_back(m);
        }

        impl_->host_bvh = build<host_bvh_type>(
                impl_->triangles.data(),
                impl_->triangles.size(),
                impl_->state->data_var == Static /* consider spatial splits if scene is static */
                );
        impl_->outlines.init(impl_->host_bvh);
    }


    // Camera matrices, viewport, render target resize

    impl_->update_viewing_params();


    // Kernel params

    aligned_vector<host_bvh_type::bvh_ref> host_primitives;
    host_primitives.push_back(impl_->host_bvh.ref());

    aligned_vector<point_light<float>> lights;

    auto add_light = [&](vec4 lpos, vec4 ldiff, float const_att, float linear_att, float quad_att)
    {
        // map OpenGL [-1,1] to Visionaray [0,1]
        ldiff += 1.0f;
        ldiff /= 2.0f;

        point_light<float> light;
        light.set_position( lpos.xyz() );
        light.set_cl( ldiff.xyz() );
        light.set_kl( ldiff.w );

        light.set_constant_attenuation(const_att);
        light.set_linear_attenuation(linear_att);
        light.set_quadratic_attenuation(quad_att);

        lights.push_back(light);
    };

    auto renderer = dynamic_cast<osgViewer::Renderer*>(opencover::coVRConfig::instance()->channels[0].camera->getRenderer());
    auto scene_view = renderer->getSceneView(0);
    auto stateset = scene_view->getGlobalStateSet();
    auto light_model = dynamic_cast<osg::LightModel*>(stateset->getAttribute(osg::StateAttribute::LIGHTMODEL));
    auto ambient = osg_cast(light_model->getAmbientIntensity());

    if (opencover::coVRLighting::instance()->headlightState)
    {
        auto headlight = scene_view->getLight();

        auto hlpos  = inverse(impl_->view_matrix) * vec4(osg_cast(headlight->getPosition()).xyz(), 1.0f);
        auto hldiff = osg_cast(headlight->getDiffuse());

        add_light(
                hlpos,
                hldiff,
                headlight->getConstantAttenuation(),
                headlight->getLinearAttenuation(),
                headlight->getQuadraticAttenuation()
            );
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

            add_light(
                    lpos,
                    ldiff,
                    l->getConstantAttenuation(),
                    l->getLinearAttenuation(),
                    l->getQuadraticAttenuation()
                );
        }
    }

    auto bounds     = impl_->host_bvh.node(0).bbox;
    auto diagonal   = bounds.max - bounds.min;
    auto epsilon    = max( 1E-3f, length(diagonal) * 1E-5f );

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
        epsilon,
        vec4(0.0f),
        impl_->state->algo == Pathtracing ? vec4(1.0f) : ambient
    );


    // Render
    impl_->call_kernel(kparams);

    impl_->host_rt.display_color_buffer();

    if (impl_->dev_state->debug_mode && impl_->dev_state->show_bvh)
    {
        glDisable(GL_LIGHTING);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadMatrixf(impl_->proj_matrix.data());

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadMatrixf(impl_->view_matrix.data());

        glColor3f(1.0f, 1.0f, 1.0f);
        impl_->outlines.frame();

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
    }

    impl_->restore_gl_state();
}

}} // namespace visionaray::cover
