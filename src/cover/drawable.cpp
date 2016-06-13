// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>

#include <GL/glew.h>

#include <osg/io_utils>
#include <osg/LightModel>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Sequence>
#include <osg/Switch>
#include <osg/TriangleIndexFunctor>

#include <osgViewer/Renderer>

#include <cover/coVRConfig.h>
#include <cover/coVRLighting.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/VRViewer.h>

#include <visionaray/detail/call_kernel.h>
#include <visionaray/detail/render_bvh.h>
#include <visionaray/gl/debug_callback.h>
#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/bvh.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/generic_material.h>
#include <visionaray/kernels.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>

#ifdef __CUDACC__
#include <visionaray/pixel_unpack_buffer_rt.h>
#endif

#include "kernels/bvh_costs_kernel.h"
#include "kernels/normals_kernel.h"
#include "kernels/tex_coords_kernel.h"
#include "drawable.h"
#include "state.h"

namespace visionaray { namespace cover {


//-------------------------------------------------------------------------------------------------
// Type definitions
//

using triangle_type             = basic_triangle<3, float>;
using triangle_list             = aligned_vector<triangle_type>;
using normal_list               = aligned_vector<vec3>;
using tex_coord_list            = aligned_vector<vec2>;
using material_type             = generic_material<matte<float>, plastic<float>, emissive<float>>;
using material_list             = aligned_vector<material_type>;
using color_type                = vector<3, float>;
using color_list                = aligned_vector<color_type>;

using host_tex_type             = texture<vector<4, unorm<8>>, NormalizedFloat, 2>;
using host_tex_ref_type         = typename host_tex_type::ref_type;
using texture_list              = aligned_vector<host_tex_ref_type>;
using texture_map               = std::map<std::string, host_tex_type>;

using host_ray_type             = basic_ray<simd::float4>;
using host_bvh_type             = index_bvh<triangle_type>;
using host_render_target_type   = cpu_buffer_rt<PF_RGBA32F, PF_DEPTH24_STENCIL8>;
using host_sched_type           = tiled_sched<host_ray_type>;

#ifdef __CUDACC__
using device_tex_type           = cuda_texture<vector<4, unorm<8>>, NormalizedFloat, 2>;
using device_tex_ref_type       = typename device_tex_type::ref_type;
using device_texture_list       = thrust::device_vector<device_tex_ref_type>;
using device_texture_map        = std::map<std::string, device_tex_type>;
using device_ray_type           = basic_ray<float>;
using device_bvh_type           = cuda_index_bvh<triangle_type>;
using device_render_target_type = pixel_unpack_buffer_rt<PF_RGBA32F, PF_DEPTH24_STENCIL8>;
using device_sched_type         = cuda_sched<device_ray_type>;
#endif


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

tex_filter_mode osg_cast(osg::Texture::FilterMode mode)
{
    switch (mode)
    {

    default:
        // fall-through
    case osg::Texture::LINEAR:
    case osg::Texture::LINEAR_MIPMAP_LINEAR:
    case osg::Texture::LINEAR_MIPMAP_NEAREST:
        return visionaray::Linear;

    case osg::Texture::NEAREST:
    case osg::Texture::NEAREST_MIPMAP_LINEAR:
    case osg::Texture::NEAREST_MIPMAP_NEAREST:
        return visionaray::Nearest;

    }
}


//-------------------------------------------------------------------------------------------------
// Get stereo mode from osg::RenderInfo
//

osg::DisplaySettings::StereoMode get_stereo_mode(osg::RenderInfo const& info)
{
    if (auto state = info.getState())
    {
        if (auto ds = state->getDisplaySettings())
        {
            return ds->getStereoMode();
        }
    }

    // StereoMode unfortunately provides no reasonable default
    return osg::DisplaySettings::StereoMode();
}


//-------------------------------------------------------------------------------------------------
// Functor that stores triangles from osg::Drawable
//

class store_triangle
{
public:

    void init(
            osg::Vec3Array const*   in_vertices,
            osg::Vec3Array const*   in_normals,
            osg::Vec4Array const*   in_colors,
            osg::Vec2Array const*   in_tex_coords,
            osg::Matrix const&      in_trans_mat,
            unsigned                in_geom_id,
            triangle_list&          out_triangles,
            normal_list&            out_normals,
            color_list&             out_colors,
            tex_coord_list&         out_tex_coords
            )
    {
        in.vertices     = in_vertices;
        in.normals      = in_normals;
        in.colors       = in_colors;
        in.tex_coords   = in_tex_coords;
        in.trans_mat    = in_trans_mat;
        in.geom_id      = in_geom_id;
        out.triangles   = &out_triangles;
        out.normals     = &out_normals;
        out.colors      = &out_colors;
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

        tri.v1 = osg_cast(v1);
        tri.e1 = osg_cast(v2) - tri.v1;
        tri.e2 = osg_cast(v3) - tri.v1;

        if (length(cross(tri.e1, tri.e2)) == 0.0f)
        {
            // TODO: implement some kind of error logging
            return;
        }

        tri.prim_id = static_cast<unsigned>(out.triangles->size());
        tri.geom_id = in.geom_id;
        out.triangles->push_back(tri);


        // normals

        assert( in.normals && out.normals );
        assert( in.normals->size() > i1 && in.normals->size() > i2 && in.normals->size() > i3 );


        auto inv_trans_mat = osg::Matrix::inverse(in.trans_mat);

        // mul left instead of transposing the matrix
        // see http://forum.openscenegraph.org/viewtopic.php?t=2494
        auto osg_n1 = inv_trans_mat * osg::Vec4((*in.normals)[i1], 1.0);
        auto osg_n2 = inv_trans_mat * osg::Vec4((*in.normals)[i2], 1.0);
        auto osg_n3 = inv_trans_mat * osg::Vec4((*in.normals)[i3], 1.0);

        auto n1 = osg_cast(osg_n1).xyz();
        auto n2 = osg_cast(osg_n2).xyz();
        auto n3 = osg_cast(osg_n3).xyz();

        // assign normalize(vec3(1, 1, 1)) to zero-length normals
        auto validate = [=](vec3 n) -> vec3
        {
            vec3 result = n;
            if (length(result) < numeric_limits<float>::epsilon())
            {
                result = vec3(1.0f);
            }
            return normalize(result);
        };

        out.normals->push_back(validate(n1));
        out.normals->push_back(validate(n2));
        out.normals->push_back(validate(n3));

        assert( out.triangles->size() == out.normals->size() / 3 );


        // colors

        if (in.colors && in.colors->getBinding() == osg::Array::BIND_PER_VERTEX && in.colors->size() > max(i1, i2, i3) )
        {
            out.colors->push_back(osg_cast((*in.colors)[i1]).xyz());
            out.colors->push_back(osg_cast((*in.colors)[i2]).xyz());
            out.colors->push_back(osg_cast((*in.colors)[i3]).xyz());
        }
        else if (in.colors && in.colors->getBinding() == osg::Array::BIND_OVERALL && in.colors->size() >= 1)
        {
            out.colors->push_back(osg_cast((*in.colors)[0]).xyz());
            out.colors->push_back(osg_cast((*in.colors)[0]).xyz());
            out.colors->push_back(osg_cast((*in.colors)[0]).xyz());
        }
        else
        {
            out.colors->emplace_back(1.0f);
            out.colors->emplace_back(1.0f);
            out.colors->emplace_back(1.0f);
        }


        // tex coords

        if ( in.tex_coords && in.tex_coords->size() > max(i1, i2, i3) )
        {
            out.tex_coords->push_back( osg_cast((*in.tex_coords)[i1]) );
            out.tex_coords->push_back( osg_cast((*in.tex_coords)[i2]) );
            out.tex_coords->push_back( osg_cast((*in.tex_coords)[i3]) );
        }
        else
        {
            out.tex_coords->emplace_back(0.0f);
            out.tex_coords->emplace_back(0.0f);
            out.tex_coords->emplace_back(0.0f);
        }
    }

private:

    // Store pointers because osg::TriangleIndexFunctor is shitty..

    struct
    {
        osg::Vec3Array const*   vertices    = nullptr;
        osg::Vec3Array const*   normals     = nullptr;
        osg::Vec4Array const*   colors      = nullptr;
        osg::Vec2Array const*   tex_coords  = nullptr;
        osg::Matrix             trans_mat;
        unsigned                geom_id;
    } in;

    struct
    {
        triangle_list*          triangles   = nullptr;
        normal_list*            normals     = nullptr;
        color_list*             colors      = nullptr;
        tex_coord_list*         tex_coords  = nullptr;
    } out;

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

        auto seq = dynamic_cast<osg::Sequence*>(&node);
        if (seq && seq->containsNode(last_child_))
        {
            auto ts = seq->getValue();
            visible_ &= seq->getChild(ts) == last_child_;
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

    get_scene_visitor(
            triangle_list&  triangles,
            normal_list&    normals,
            color_list&     colors,
            tex_coord_list& tex_coords,
            material_list&  materials,
            texture_map&    textures,
            texture_list&   texture_refs,
            TraversalMode   tm
            )
        : base_type(tm)
        , triangles_(triangles)
        , normals_(normals)
        , colors_(colors)
        , tex_coords_(tex_coords)
        , materials_(materials)
        , textures_(textures)
        , texture_refs_(texture_refs)
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

            auto node_colors = dynamic_cast<osg::Vec4Array*>(geom->getColorArray());
            // ok if node_colors == 0


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
                auto ce = mat->getEmission(osg::Material::Face::FRONT);

                if (ce[0] > 0.0f || ce[1] > 0.0f || ce[2] > 0.0f)
                {
                    emissive<float> vsnray_mat;
                    vsnray_mat.set_ce( from_rgb(osg_cast(ce).xyz()) );
                    vsnray_mat.set_ls( 1.0f );
                    materials_.push_back(vsnray_mat);
                }
                else if (cs[0] == 0.0f && cs[1] == 0.0f && cs[2] == 0.0f)
                {
                    matte<float> vsnray_mat;
                    vsnray_mat.set_ca( from_rgb(osg_cast(ca).xyz()) );
                    vsnray_mat.set_cd( from_rgb(osg_cast(cd).xyz()) );
                    vsnray_mat.set_ka( 1.0f );
                    vsnray_mat.set_kd( 1.0f );
                    materials_.push_back(vsnray_mat);
                }
                else
                {
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
            }
            else
            {
                plastic<float> vsnray_mat;
                vsnray_mat.set_ca( from_rgb(0.2f, 0.2f, 0.2f) );
                vsnray_mat.set_cd( from_rgb(0.8f, 0.8f, 0.8f) );
                vsnray_mat.set_cs( from_rgb(0.1f, 0.1f, 0.1f) );
                vsnray_mat.set_ka( 1.0f );
                vsnray_mat.set_kd( 1.0f );
                vsnray_mat.set_ks( 1.0f );
                vsnray_mat.set_specular_exp( 32.0f );
                materials_.push_back(vsnray_mat);
            }


            // texture

            auto tattr = set->getTextureAttribute(0, osg::StateAttribute::TEXTURE);
            auto tex = dynamic_cast<osg::Texture2D*>(tattr);
            auto img = tex != nullptr ? tex->getImage() : nullptr;

            if (tex && img)
            {
                assert( img->isDataContiguous() ); // TODO

                auto dest_format = PF_RGBA8;
                auto source_format = map_gl_format(
                        img->getPixelFormat(),
                        img->getDataType(),
                        osg::Image::computeNumComponents(img->getPixelFormat()) * sizeof(uint8_t) /* TODO */
                        );

                auto source_info = map_pixel_format(source_format);

                assert( source_info.components == 3 || source_info.components == 4 );

                std::string filename = img->getFileName();

                if (filename.empty())
                {
                    filename = std::string("TEXTURE") + std::to_string( textures_.size() );
                }

                auto p = textures_.emplace( std::make_pair(
                        filename,
                        host_tex_type(img->s(), img->t())
                        ) );

                bool inserted = p.second;
                auto it = inserted ? p.first : textures_.find( img->getFileName() );
                assert( it != textures_.end() );

                auto& vsnray_tex = it->second;

                if (inserted)
                {
                    vsnray_tex.set_address_mode( 0, osg_cast(tex->getWrap(osg::Texture::WRAP_S)) );
                    vsnray_tex.set_address_mode( 1, osg_cast(tex->getWrap(osg::Texture::WRAP_T)) );

//                  auto min_filter = tex->getFilter(osg::Texture::MIN_FILTER);
                    auto mag_filter = tex->getFilter(osg::Texture::MAG_FILTER);

                    vsnray_tex.set_filter_mode( osg_cast(mag_filter) );

                    if (source_info.components == 3)
                    {
                        auto data_ptr = reinterpret_cast<vector<3, unorm<8>> const*>(img->data());
                        vsnray_tex.set_data(data_ptr, source_format, dest_format);
                    }
                    else if (source_info.components == 4)
                    {
                        auto data_ptr = reinterpret_cast<vector<4, unorm<8>> const*>(img->data());
                        vsnray_tex.set_data(data_ptr, source_format, dest_format);
                    }
                    else
                    {
                        assert(0); // TODO
                    }
                }

                texture_refs_.emplace_back( vsnray_tex );
            }
            else
            {
                texture_refs_.emplace_back( 0, 0 );
            }

            assert( materials_.size() == texture_refs_.size() );


            // transform

            auto world_transform = osg::computeLocalToWorld(getNodePath());


            // geometry

            assert( static_cast<material_list::size_type>(static_cast<unsigned>(materials_.size()) == materials_.size()) );
            unsigned geom_id = static_cast<unsigned>(materials_.size() - 1);

            osg::TriangleIndexFunctor<store_triangle> tif;
            tif.init(
                    node_vertices,
                    node_normals,
                    node_colors,
                    node_tex_coords,
                    world_transform,
                    geom_id,
                    triangles_,
                    normals_,
                    colors_,
                    tex_coords_
                    );
            geom->accept(tif);
        }

        base_type::traverse(geode);
    }

private:

    triangle_list&  triangles_;
    normal_list&    normals_;
    color_list&     colors_;
    tex_coord_list& tex_coords_;
    material_list&  materials_;
    texture_map&    textures_;
    texture_list&   texture_refs_;

};


//-------------------------------------------------------------------------------------------------
// TODO: use make_intersector(lambda...) instead
//

template <typename Texture>
struct mask_intersector : basic_intersector<mask_intersector<Texture>>
{
    using basic_intersector<mask_intersector<Texture>>::operator();

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(R const& ray, basic_triangle<3, S> const& tri)
        -> decltype( intersect(ray, tri) )
    {
        auto hr = intersect(ray, tri);

        if ( !any(hr.hit) )
        {
            return hr;
        }

        auto tex_color = get_tex_color(hr); 
        hr.hit &= tex_color.w >= S(0.01);
 
        return hr;
    }

    vec2 const*                 tex_coords;
    Texture const*              textures;

private:

    template <typename HR>
    VSNRAY_FUNC
    vector<4, float> get_tex_color(HR const& hr)
    {
        auto tc = get_tex_coord(tex_coords, hr);
        auto const& tex = textures[hr.geom_id];
        return tex.width() > 0 && tex.height() > 0
                       ? vector<4, float>(tex2D(tex, tc))
                       : vector<4, float>(1.0);
    }

    template <
        typename T,
        typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
        >
    VSNRAY_CPU_FUNC
    vector<4, T> get_tex_color(hit_record<basic_ray<T>, primitive<unsigned>> const& hr)
    {
        auto tc = get_tex_coord(tex_coords, hr);

        auto hrs = unpack( hr );
        auto tcs = unpack( tc );

        std::array<vector<4, float>, simd::num_elements<T>::value> tex_colors;

        for (unsigned i = 0; i < simd::num_elements<T>::value; ++i)
        {
            if (!hrs[i].hit)
            {
                continue;
            }

            auto const& tex = textures[hrs[i].geom_id];
            tex_colors[i] = tex.width() > 0 && tex.height() > 0
                          ? vector<4, float>(tex2D(tex, tcs[i]))
                          : vector<4, float>(1.0);
        }

        return simd::pack(tex_colors);
    }
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

    triangle_list                           triangles;
    normal_list                             normals;
    tex_coord_list                          tex_coords;
    material_list                           materials;
    color_list                              colors;
    texture_map                             textures;
    texture_list                            texture_refs;
    host_bvh_type                           host_bvh;
    host_sched_type                         host_sched;
    mask_intersector<host_tex_ref_type>     host_intersector;

#ifdef __CUDACC__
    thrust::device_vector<vec3>             device_normals;
    thrust::device_vector<vec2>             device_tex_coords;
    thrust::device_vector<material_type>    device_materials;
    thrust::device_vector<color_type>       device_colors;
    device_texture_map                      device_textures;
    device_texture_list                     device_texture_refs;
    device_bvh_type                         device_bvh;
    device_sched_type                       device_sched;
    mask_intersector<device_tex_ref_type>   device_intersector;
#endif

    enum eye
    {
        Left,
        Right
    };

    struct viewing_params
    {
        host_render_target_type             host_rt;
#ifdef __CUDACC__
        device_render_target_type           device_rt;
#endif
        mat4                                view_matrix;
        mat4                                proj_matrix;
        recti                               viewport;
        unsigned                            frame_num       = 0;
    };

    viewing_params                          eye_params[2]; // for left and right eye

    eye                                     current_eye     = Right;

    size_t                                  total_frame_num = 0;

    color_space                             clr_space       = RGB;
    detail::algorithm                       algo_current    = detail::Simple;
    unsigned                                num_bounces     = 4;
    device_type                             device          = CPU;

    detail::bvh_outline_renderer            outlines;

    gl::debug_callback                      gl_debug_callback;

    bool                                    glew_init       = false;

    std::shared_ptr<render_state>           state           = nullptr;
    std::shared_ptr<debug_state>            dev_state       = nullptr;
    struct
    {
        GLint                               matrix_mode;
        GLboolean                           lighting;
        GLboolean                           depth_test;
        GLboolean                           framebuffer_srgb;
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
                  : std::thread::hardware_concurrency()
                    );
        }
    }

    void store_gl_state();
    void restore_gl_state();
    void update_viewing_params(osg::DisplaySettings::StereoMode mode);
    void update_device_data();
    void commit_state();
    
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
    gl_state.depth_test = glIsEnabled(GL_DEPTH_TEST);
    gl_state.framebuffer_srgb = glIsEnabled(GL_FRAMEBUFFER_SRGB);
}

void drawable::impl::restore_gl_state()
{
    if (gl_state.framebuffer_srgb)
    {
        glEnable(GL_FRAMEBUFFER_SRGB);
    }
    else
    {
        glDisable(GL_FRAMEBUFFER_SRGB);
    }

    if (gl_state.depth_test)
    {
        glEnable(GL_DEPTH_TEST);
    }
    else
    {
        glDisable(GL_DEPTH_TEST);
    }

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

void drawable::impl::update_viewing_params(osg::DisplaySettings::StereoMode mode)
{
    current_eye = Right; // default if no stereo

    if (opencover::coVRConfig::instance()->stereoState())
    {
        switch (mode)
        {
        // TODO: implement remaining modes
        case osg::DisplaySettings::ANAGLYPHIC:
            {
                if (total_frame_num % 2 == 1)
                {
                    current_eye = Left;
                }
            }
            break;
        case osg::DisplaySettings::QUAD_BUFFER:
            {
                GLint db = 0;
                glGetIntegerv(GL_DRAW_BUFFER, &db);
                if (db != GL_BACK_RIGHT && db != GL_FRONT_RIGHT && db != GL_RIGHT)
                {
                    current_eye = Left;
                }
            }
            break;
        default:
            break;
        }
    }

    auto osg_cam = opencover::coVRConfig::instance()->channels[0].camera;

    // Matrices

    auto t = opencover::cover->getXformMat();
    auto s = opencover::cover->getObjectsScale()->getMatrix();
    auto v = current_eye == Right
            ? opencover::coVRConfig::instance()->channels[0].rightView
            : opencover::coVRConfig::instance()->channels[0].leftView
            ;
    auto view = osg_cast( s * t * v );
    auto proj = current_eye == Right
            ? osg_cast( opencover::coVRConfig::instance()->channels[0].rightProj )
            : osg_cast( opencover::coVRConfig::instance()->channels[0].leftProj )
            ;


    // Viewport

    auto osg_viewport = osg_cam->getViewport();
    recti vp(osg_viewport->x(), osg_viewport->y(), osg_viewport->width(), osg_viewport->height());


    // Reset frame counter on change or if scene is dynamic

    auto& vparams = eye_params[current_eye];

    if (
        state->data_var     == Dynamic      ||
        state->clr_space    != clr_space    ||
        state->algo         != algo_current ||
        state->device       != device       ||
        state->num_bounces  != num_bounces
        )
    {
        eye_params[Left ].frame_num = 0;
        eye_params[Right].frame_num = 0;
    }

    if (
        vparams.view_matrix != view         ||
        vparams.proj_matrix != proj         ||
        vparams.viewport    != vp
        )
    {
        vparams.frame_num = 0;
    }


    // Update

    ++total_frame_num;

    vparams.view_matrix  = view;
    vparams.proj_matrix  = proj;

    if (vparams.viewport != vp)
    {
        vparams.viewport = vp;
        vparams.host_rt.resize(vparams.viewport[2], vparams.viewport[3]);
#ifdef __CUDACC__
        vparams.device_rt.resize(vparams.viewport[2], vparams.viewport[3]);
#endif
    }
}

void drawable::impl::update_device_data()
{
#ifdef __CUDACC__
    if ( state->device == GPU && (state->data_var == Dynamic || state->device != device) )
    {
        device_bvh          = device_bvh_type(host_bvh);
        device_normals      = normals;
        device_colors       = colors;
        device_tex_coords   = tex_coords;
        device_materials    = materials;

        device_textures.clear();
        device_texture_refs.clear();

        device_texture_refs.resize(texture_refs.size());

        for (auto const& pair_host_tex : textures)
        {
            auto const& host_tex = pair_host_tex.second;
            device_tex_type device_tex(pair_host_tex.second);
            auto const& p = device_textures.emplace( pair_host_tex.first, std::move(device_tex) );

            assert( p.second /* inserted */ );

            auto it = p.first;

            // TODO: construct GPU data during get_scene_visitor traversal
            for (size_t i = 0; i < texture_refs.size(); ++i)
            {
                if ( texture_refs[i].data() == host_tex.data() )
                {
                    device_texture_refs[i] = device_tex_ref_type(it->second);
                }
            }
        }
    }
#endif
}

void drawable::impl::commit_state()
{
    clr_space       = state->clr_space;
    algo_current    = state->algo;
    num_bounces     = state->num_bounces;
    device          = state->device;
}


//-------------------------------------------------------------------------------------------------
// Call either one of the visionaray kernels or a custom one
//

template <typename KParams>
void drawable::impl::call_kernel(KParams const& params)
{
    if (dev_state->debug_mode && (dev_state->show_bvh_costs || dev_state->show_normals || dev_state->show_tex_coords))
    {
        call_kernel_debug( params );
    }
    else
    {
        auto& vparams = eye_params[current_eye];
        if (state->device == GPU)
        {
#ifdef __CUDACC__
            visionaray::detail::call_kernel(
                    state->algo,
                    device_sched,
                    params,
                    vparams.frame_num,
                    vparams.view_matrix,
                    vparams.proj_matrix,
                    vparams.viewport,
                    vparams.device_rt,
                    device_intersector
                    );
#endif
        }
        else
        {
#ifndef __CUDA_ARCH__
            visionaray::detail::call_kernel(
                    state->algo,
                    host_sched,
                    params,
                    vparams.frame_num,
                    vparams.view_matrix,
                    vparams.proj_matrix,
                    vparams.viewport,
                    vparams.host_rt,
                    host_intersector
                    );
#endif
        }
    }
}


//-------------------------------------------------------------------------------------------------
// Custom kernels to debug some internal values
//

template <typename KParams>
void drawable::impl::call_kernel_debug(KParams const& params)
{
    if (state->device == GPU)
    {
#ifdef __CUDACC__
        auto& vparams = eye_params[current_eye];

        auto sparams = make_sched_params(
                vparams.view_matrix,
                vparams.proj_matrix,
                vparams.viewport,
                vparams.device_rt
                );

        if (dev_state->show_bvh_costs)
        {
            bvh_costs_kernel<KParams> k(params);
            device_sched.frame(k, sparams);
        }
        else if (dev_state->show_normals)
        {
            normals_kernel<KParams> k(params);
            device_sched.frame(k, sparams);
        }
        else if (dev_state->show_tex_coords)
        {
            tex_coords_kernel<KParams> k(params);
            device_sched.frame(k, sparams);
        }
#endif // __CUDACC__
    }
    else if (state->device == CPU)
    {
#ifndef __CUDA_ARCH__
        auto& vparams = eye_params[current_eye];

        auto sparams = make_sched_params(
                vparams.view_matrix,
                vparams.proj_matrix,
                vparams.viewport,
                vparams.host_rt
                );

        if (dev_state->show_bvh_costs)
        {
            bvh_costs_kernel<KParams> k(params);
            host_sched.frame(k, sparams);
        }
        else if (dev_state->show_normals)
        {
            normals_kernel<KParams> k(params);
            host_sched.frame(k, sparams);
        }
        else if (dev_state->show_tex_coords)
        {
            tex_coords_kernel<KParams> k(params);
            host_sched.frame(k, sparams);
        }
#endif // __CUDA_ARCH__
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
    impl_->outlines.destroy();
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

        bounds = combine(bounds, v1);
        bounds = combine(bounds, v2);
        bounds = combine(bounds, v3);
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

void drawable::drawImplementation(osg::RenderInfo& info) const
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


    // Activate debug callback

    gl::debug_params params;
    if (opencover::cover->debugLevel(4))
    {
        params.level = gl::debug_level::Notification;
    }
    else if (opencover::cover->debugLevel(2))
    {
        params.level = gl::debug_level::Low;
    }
    else if (opencover::cover->debugLevel(1))
    {
        params.level = gl::debug_level::Medium;
    }
    else if (opencover::cover->debugLevel(0))
    {
        params.level = gl::debug_level::High;
    }
    impl_->gl_debug_callback.activate(params);

    impl_->store_gl_state();


    // Scene data

    if (impl_->state->data_var == Dynamic || impl_->triangles.size() == 0)
    {
        // TODO: real dynamic scenes :)

        impl_->triangles.clear();
        impl_->normals.clear();
        impl_->colors.clear();
        impl_->tex_coords.clear();
        impl_->materials.clear();
        impl_->texture_refs.clear();

        get_scene_visitor visitor(
                impl_->triangles,
                impl_->normals,
                impl_->colors,
                impl_->tex_coords,
                impl_->materials,
                impl_->textures,
                impl_->texture_refs,
                osg::NodeVisitor::TRAVERSE_ALL_CHILDREN
                );
        opencover::cover->getObjectsRoot()->accept(visitor);

        if (impl_->triangles.size() == 0)
        {
            return;
        }

        impl_->host_bvh = build<host_bvh_type>(
                impl_->triangles.data(),
                impl_->triangles.size(),
                impl_->state->data_var == Static /* consider spatial splits if scene is static */
                );
        impl_->outlines.init(impl_->host_bvh);
    }


    // Camera matrices, viewport, render target resize

    impl_->update_viewing_params(get_stereo_mode(info));

    // Copy BVH, normals, etc. if necessary

    impl_->update_device_data();

    // Finally update state variables. Call after any other updates!

    impl_->commit_state();


    // Kernel params

    aligned_vector<host_bvh_type::bvh_ref> host_primitives;
    host_primitives.push_back(impl_->host_bvh.ref());

    auto renderer = dynamic_cast<osgViewer::Renderer*>(opencover::coVRConfig::instance()->channels[0].camera->getRenderer());
    auto scene_view = renderer->getSceneView(0);
    auto stateset = scene_view->getGlobalStateSet();
    auto light_model = dynamic_cast<osg::LightModel*>(stateset->getAttribute(osg::StateAttribute::LIGHTMODEL));
    auto ambient = osg_cast(light_model->getAmbientIntensity());


    using light_type = point_light<float>;

    aligned_vector<light_type> lights;

    auto cover_lights = opencover::coVRLighting::instance()->lightList;
    for (auto it = cover_lights.begin(); it != cover_lights.end(); ++it)
    {
        if ((*it).on)
        {
            auto l = (*it).source->getLight();

            // TODO
            auto lpos  = osg_cast(l->getPosition());
            auto ldiff = osg_cast(l->getDiffuse());

            if ((*it).source->getParent(0) == opencover::VRSceneGraph::instance()->getScene())
            {
                // Light source is fixed to scene (e.g. headlight)
                auto trans = osg::computeLocalToWorld(opencover::cover->getObjectsRoot()->getParentalNodePaths()[0]);
                lpos = inverse(osg_cast(trans)) * lpos;
            }

                    // map OpenGL [-1,1] to Visionaray [0,1]
            ldiff += 1.0f;
            ldiff /= 2.0f;

            point_light<float> light;
            light.set_position( lpos.xyz() );
            light.set_cl( ldiff.xyz() );
            light.set_kl( ldiff.w );

            light.set_constant_attenuation(l->getConstantAttenuation());
            light.set_linear_attenuation(l->getLinearAttenuation());
            light.set_quadratic_attenuation(l->getQuadraticAttenuation());

            lights.push_back(light);
        }
    }

    auto bounds     = impl_->host_bvh.node(0).bbox;
    auto diagonal   = bounds.max - bounds.min;
    auto bounces    = impl_->state->num_bounces;
    auto epsilon    = max( 1E-3f, length(diagonal) * 1E-5f );


    if (impl_->state->clr_space == sRGB)
    {
        glEnable(GL_FRAMEBUFFER_SRGB);
    }
    else
    {
        glDisable(GL_FRAMEBUFFER_SRGB);
    }

    auto& vparams = impl_->eye_params[impl_->current_eye];

    if (impl_->state->device == GPU)
    {
#ifdef __CUDACC__
        thrust::device_vector<device_bvh_type::bvh_ref> device_primitives;

        device_primitives.push_back(impl_->device_bvh.ref());

        thrust::device_vector<light_type> device_lights = lights;

        auto kparams = make_kernel_params(
                normals_per_vertex_binding{},
                colors_per_vertex_binding{},
                thrust::raw_pointer_cast(device_primitives.data()),
                thrust::raw_pointer_cast(device_primitives.data()) + device_primitives.size(),
                thrust::raw_pointer_cast(impl_->device_normals.data()),
                thrust::raw_pointer_cast(impl_->device_tex_coords.data()),
                thrust::raw_pointer_cast(impl_->device_materials.data()),
                thrust::raw_pointer_cast(impl_->device_colors.data()),
                thrust::raw_pointer_cast(impl_->device_texture_refs.data()),
                thrust::raw_pointer_cast(device_lights.data()),
                thrust::raw_pointer_cast(device_lights.data()) + device_lights.size(),
                bounces,
                epsilon,
                vec4(0.0f),
                impl_->state->algo == detail::Pathtracing ? vec4(1.0f) : ambient
                );

        impl_->device_intersector.tex_coords = kparams.tex_coords;
        impl_->device_intersector.textures   = kparams.textures;

        impl_->call_kernel(kparams);

        vparams.device_rt.display_color_buffer();
#endif
    }
    else if (impl_->state->device == CPU)
    {
#ifndef __CUDA_ARCH__
        auto kparams = make_kernel_params(
                normals_per_vertex_binding{},
                colors_per_vertex_binding{},
                host_primitives.data(),
                host_primitives.data() + host_primitives.size(),
                impl_->normals.data(),
                impl_->tex_coords.data(),
                impl_->materials.data(),
                impl_->colors.data(),
                impl_->texture_refs.data(),
                lights.data(),
                lights.data() + lights.size(),
                bounces,
                epsilon,
                vec4(0.0f),
                impl_->state->algo == detail::Pathtracing ? vec4(1.0f) : ambient
                );

        impl_->host_intersector.tex_coords = kparams.tex_coords;
        impl_->host_intersector.textures   = kparams.textures;

        impl_->call_kernel(kparams);

        vparams.host_rt.display_color_buffer();
#endif
    }

    if (impl_->dev_state->debug_mode && impl_->dev_state->show_bvh)
    {
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadMatrixf(vparams.proj_matrix.data());

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadMatrixf(vparams.view_matrix.data());

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
