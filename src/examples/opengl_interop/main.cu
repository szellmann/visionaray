// This file is distributed under the MIT license.
// See the LICENSE file for details.

//-------------------------------------------------------------------------------------------------
// This file is based on Peter Shirley's book "Ray Tracing in One Weekend"
//

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <memory>

#include <GL/glew.h>

#include <cuda_runtime_api.h>

#include <thrust/device_vector.h>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/detail/platform.h>

#include <visionaray/gl/debug_callback.h>
#include <visionaray/gl/handle.h>
#include <visionaray/gl/program.h>
#include <visionaray/gl/shader.h>

#include <visionaray/bvh.h>
#include <visionaray/pixel_unpack_buffer_rt.h>
#include <visionaray/generic_material.h>
#include <visionaray/kernels.h>
#include <visionaray/material.h>
#include <visionaray/matrix_camera.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/png_image.h>
#include <common/viewer_glut.h>

#ifdef _WIN32

//-------------------------------------------------------------------------------------------------
// https://pubs.opengroup.org/onlinepubs/007908799/xsh/drand48.html
//

double drand48()
{
    constexpr static uint64_t m = 1ULL<<48;
    constexpr static uint64_t a = 0x5DEECE66DULL;
    constexpr static uint64_t c = 0xBULL;
    thread_local static uint64_t x = 0;

    x = (a * x + c) & (m - 1ULL);

    return static_cast<double>(x) / m;
}

#endif

using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
//
//

struct renderer : viewer_type
{
    using ray_type = basic_ray<float>;

    renderer()
        : viewer_type(512, 512, "Visionaray OpenGL Interop Example")
        , bbox({ -3.0f, -3.0f, -3.0f }, { 3.0f, 3.0f, 3.0f })
        , device_sched(16, 16)
    {
        using namespace support;

        add_cmdline_option( cl::makeOption<unsigned&>(
            cl::Parser<>(),
            "spp",
            cl::Desc("Pixels per sample for path tracing"),
            cl::ArgRequired,
            cl::init(this->spp)
            ) );

        random_scene();

        set_background_color(vec3(0.5f, 0.7f, 1.0f));
    }

    aabb                                                    bbox;
    pinhole_camera                                          cam;
    pixel_unpack_buffer_rt<PF_RGBA32F, PF_DEPTH24_STENCIL8> device_rt; // Render target with 24-bit depth buffer
    cuda_sched<ray_type>                                    device_sched;

    unsigned                                                frame_num   = 0;
    unsigned                                                spp         = 1;

    // rendering data

    index_bvh<basic_sphere<float>>                          sphere_bvh;
    std::vector<basic_sphere<float>>                        list;
    std::vector<generic_material<
            glass<float>,
            matte<float>,
            mirror<float>
            >>                                              materials;

    // OpenGL rendering data
    std::vector<vec3>                                       centers;
    std::vector<vec3>                                       offsets;
    std::vector<vec3>                                       normals;
    std::vector<vec3>                                       albedo;
    std::vector<vec2>                                       texcoords;
    std::vector<unsigned>                                   indices;

    // copies that are located on the device
    // (we build up the initial data structures on the host!)

    cuda_index_bvh<basic_sphere<float>>                     device_bvh;
    thrust::device_vector<generic_material<
            glass<float>,
            matte<float>,
            mirror<float>
            >>                                              device_materials;

    basic_sphere<float> make_sphere(vec3 center, float radius)
    {
        static int sphere_id = 0;
        basic_sphere<float> sphere(center, radius);
        sphere.prim_id = sphere_id;
        sphere.geom_id = sphere_id;
        ++sphere_id;
        return sphere;
    }

    glass<float> make_dielectric(float ior)
    {
        glass<float> mat;
        mat.ct() = from_rgb(1.0f, 1.0f, 1.0f);
        mat.kt() = 1.0f;
        mat.cr() = from_rgb(1.0f, 1.0f, 1.0f);
        mat.kr() = 1.0f;
        mat.ior() = spectrum<float>(ior);
        return mat;
    }

    matte<float> make_lambertian(vec3 cd)
    {
        matte<float> mat;
        mat.ca() = from_rgb(0.0f, 0.0f, 0.0f);
        mat.ka() = 0.0f;
        mat.cd() = from_rgb(cd);
        mat.kd() = 1.0f;
        return mat;
    }

    mirror<float> make_metal(vec3 cr)
    {
        mirror<float> mat;
        mat.cr() = from_rgb(cr);
        mat.kr() = 1.0f;
        mat.ior() = spectrum<float>(0.0);
        mat.absorption() = spectrum<float>(0.0);
        return mat;
    }

    void random_scene()
    {
        int n = 500;
        list.resize(n + 1);
        materials.resize(n + 1);
        list[0] = make_sphere(vec3(0, -1000, 0), 1000);
        materials[0] = make_lambertian(vec3(0.5f, 0.5f, 0.5f));
        int i = 1;
        for (int a = -11; a < 11; ++a)
        {
            for (int b = -11; b < 11; ++b)
            {
                float choose_mat = drand48();
                vec3 center(a + 0.9 * drand48(), 0.2, b + 0.9 * drand48());
                if (length(center - vec3(4, 0.2, 0)) > 0.9)
                {
                    // Add half of the primitives as ray-traced spheres,
                    // the other half as OpenGL-renderable boxes
                    if (i % 2 == 0)
                    {
                        list[i] = make_sphere(center, 0.2);
                    }
                    else
                    {
                        // We split the vertices into centers and corner
                        // offsets so that we can later apply that fancy
                        // rotation in the vertex shader

                        aabb box(vec3(-0.23), vec3(0.23));
                        auto boxverts = compute_vertices(box);
                        mat3 rot = mat3::rotation(normalize(vec3(drand48(), drand48(), drand48())), drand48());
                        for (auto& v : boxverts)
                        {
                            v = rot * v;
                        }

                        struct vertex
                        {
                            vec3 center;
                            vec3 offset;
                            vec3 normal;
                            vec3 albedo;
                            vec2 texcoord;
                        };

                        vec3 colors[6] = { { drand48(), drand48(), drand48() },
                                           { drand48(), drand48(), drand48() },
                                           { drand48(), drand48(), drand48() },
                                           { drand48(), drand48(), drand48() },
                                           { drand48(), drand48(), drand48() },
                                           { drand48(), drand48(), drand48() } };

                        vertex verts[24] = {
                            { center, boxverts[0], vec3(0,0,+1), colors[0], vec2(0,-0.21) },
                            { center, boxverts[1], vec3(0,0,+1), colors[0], vec2(1,-0.21) },
                            { center, boxverts[2], vec3(0,0,+1), colors[0], vec2(1,1.21) },
                            { center, boxverts[3], vec3(0,0,+1), colors[0], vec2(0,1.21) },

                            { center, boxverts[4], vec3(0,0,-1), colors[1], vec2(0,-0.21) },
                            { center, boxverts[5], vec3(0,0,-1), colors[1], vec2(1,-0.21) },
                            { center, boxverts[6], vec3(0,0,-1), colors[1], vec2(1,1.21) },
                            { center, boxverts[7], vec3(0,0,-1), colors[1], vec2(0,1.21) },

                            { center, boxverts[1], vec3(+1,0,0), colors[2], vec2(0,-0.21) },
                            { center, boxverts[4], vec3(+1,0,0), colors[2], vec2(1,-0.21) },
                            { center, boxverts[7], vec3(+1,0,0), colors[2], vec2(1,1.21) },
                            { center, boxverts[2], vec3(+1,0,0), colors[2], vec2(0,1.21) },

                            { center, boxverts[5], vec3(-1,0,0), colors[3], vec2(0,-0.21) },
                            { center, boxverts[0], vec3(-1,0,0), colors[3], vec2(1,-0.21) },
                            { center, boxverts[3], vec3(-1,0,0), colors[3], vec2(1,1.21) },
                            { center, boxverts[6], vec3(-1,0,0), colors[3], vec2(0,1.21) },

                            { center, boxverts[3], vec3(0,+1,0), colors[4], vec2(0,-0.21) },
                            { center, boxverts[2], vec3(0,+1,0), colors[4], vec2(1,-0.21) },
                            { center, boxverts[7], vec3(0,+1,0), colors[4], vec2(1,1.21) },
                            { center, boxverts[6], vec3(0,+1,0), colors[4], vec2(0,1.21) },

                            { center, boxverts[5], vec3(0,-1,0), colors[5], vec2(0,-0.21) },
                            { center, boxverts[0], vec3(0,-1,0), colors[5], vec2(1,-0.21) },
                            { center, boxverts[1], vec3(0,-1,0), colors[5], vec2(1,1.21) },
                            { center, boxverts[4], vec3(0,-1,0), colors[5], vec2(0,1.21) },
                            };

                        for (int j = 0; j < 24; ++j)
                        {
                            centers.push_back(verts[j].center);
                            offsets.push_back(verts[j].offset);
                            normals.push_back(verts[j].normal);
                            albedo.push_back(verts[j].albedo);
                            texcoords.push_back(verts[j].texcoord);
                        }

                        unsigned ii = static_cast<unsigned>(centers.size());
                        unsigned inds[36] = {
                                ii + 0, ii + 1, ii + 2,
                                ii + 0, ii + 2, ii + 3,

                                ii + 4, ii + 5, ii + 6,
                                ii + 4, ii + 6, ii + 7,

                                ii + 8, ii + 9, ii + 10,
                                ii + 8, ii + 10, ii + 11,

                                ii + 12, ii + 13, ii + 14,
                                ii + 12, ii + 14, ii + 15,

                                ii + 16, ii + 17, ii + 18,
                                ii + 16, ii + 18, ii + 19,

                                ii + 20, ii + 21, ii + 22,
                                ii + 20, ii + 22, ii + 23
                                };
                        indices.insert(indices.end(), inds, inds + 36);

                        // Insert a dummy sphere to keep things simple
                        list[i] = make_sphere(vec3(0.0), 0.0);
                    }

                    if (choose_mat < 0.8) // diffuse
                    {
                        materials[i] = make_lambertian(vec3(
                            static_cast<float>(drand48() * drand48()),
                            static_cast<float>(drand48() * drand48()),
                            static_cast<float>(drand48() * drand48())
                            ));
                    }
                    else if (choose_mat < 0.95) // metal
                    {
                        materials[i] = make_metal(vec3(
                            0.5f * (1.0f + static_cast<float>(drand48())),
                            0.5f * (1.0f + static_cast<float>(drand48())),
                            0.5f * (1.0f + static_cast<float>(drand48()))
                            ));
                    }
                    else
                    {
                        materials[i] = make_dielectric(1.5f);
                    }
                    ++i;
                }
            }
        }

        list[i] = make_sphere(vec3(0, 1, 0), 1.0);
        materials[i] = make_dielectric(1.5f);
        ++i;

        list[i] = make_sphere(vec3(-4, 1, 0), 1.0);
        materials[i] = make_lambertian(vec3(0.4f, 0.2f, 0.1f));
        ++i;

        list[i] = make_sphere(vec3(4, 1, 0), 1.0);
        materials[i] = make_metal(vec3(0.7f, 0.6f, 0.5f));
        ++i;

        binned_sah_builder builder;
        builder.enable_spatial_splits(true);

        sphere_bvh = builder.build(index_bvh<basic_sphere<float>>{}, list.data(), i);

        // Copy data to GPU
        device_bvh = cuda_index_bvh<basic_sphere<float>>(sphere_bvh);
        device_materials = materials;
    }


    //--- OpenGL rendering pipeline -----------------------

    struct gl_pipeline
    {
        bool        initialized = false;
        gl::buffer  center_buffer;
        gl::buffer  offset_buffer;
        gl::buffer  normal_buffer;
        gl::buffer  albedo_buffer;
        gl::buffer  texcoord_buffer;
        gl::buffer  index_buffer;
        gl::texture logo;
        gl::program prog;
        gl::shader  vert;
        gl::shader  frag;
        GLuint      center_loc;
        GLuint      offset_loc;
        GLuint      normal_loc;
        GLuint      albedo_loc;
        GLuint      texcoord_loc;
        GLuint      logo_loc;
        GLuint      view_loc;
        GLuint      proj_loc;
        GLuint      norm_loc;
        GLuint      rot_loc;
    };

    gl_pipeline pipeline;
    gl::debug_callback debugcb;

    bool init_gl()
    {
        // Store OpenGL state
        GLint array_buffer_binding = 0;
        GLint element_array_buffer_binding = 0;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &array_buffer_binding);
        glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &element_array_buffer_binding);


        // Setup shaders
        pipeline.vert.reset(glCreateShader(GL_VERTEX_SHADER));
        pipeline.vert.set_source(R"(
            attribute vec3 center;
            attribute vec3 offset;
            attribute vec3 normal;
            attribute vec3 albedo;
            attribute vec2 texcoord;

            uniform mat4 view;
            uniform mat4 proj;
            uniform mat4 norm;
            uniform mat4 rot;

            varying vec3 N;
            varying vec3 C;
            varying vec2 TC;

            void main(void)
            {
                vec4 vert = rot * vec4(offset, 1.0);
                vert.xyz += center;
                gl_Position = proj * view * vert;

                vec4 n4 = norm * vec4(normal, 1.0);
                N = n4.xyz;

                C = albedo;

                TC = texcoord;
            }
            )");
        pipeline.vert.compile();
        if (!pipeline.vert.check_compiled())
        {
            return false;
        }

        pipeline.frag.reset(glCreateShader(GL_FRAGMENT_SHADER));
        pipeline.frag.set_source(R"(
            varying vec3 N;
            varying vec3 C;
            varying vec2 TC;

            uniform sampler2D logo;

            void main(void)
            {
                vec3 N = normalize(N);
                vec3 kd = vec3(0.8) * abs(dot(N, vec3(0,0,-1))) * C * (0.6 + 0.4 * texture2D(logo, TC).w);
                gl_FragColor = vec4(kd, 1.0);
            }
            )");
        pipeline.frag.compile();
        if (!pipeline.frag.check_compiled())
        {
            return false;
        }

        pipeline.prog.reset(glCreateProgram());
        pipeline.prog.attach_shader(pipeline.vert);
        pipeline.prog.attach_shader(pipeline.frag);

        pipeline.prog.link();
        if (!pipeline.prog.check_linked())
        {
            return false;
        }

        pipeline.center_loc   = glGetAttribLocation(pipeline.prog.get(), "center");
        pipeline.offset_loc   = glGetAttribLocation(pipeline.prog.get(), "offset");
        pipeline.normal_loc   = glGetAttribLocation(pipeline.prog.get(), "normal");
        pipeline.albedo_loc   = glGetAttribLocation(pipeline.prog.get(), "albedo");
        pipeline.texcoord_loc = glGetAttribLocation(pipeline.prog.get(), "texcoord");
        pipeline.logo_loc     = glGetUniformLocation(pipeline.prog.get(), "logo");
        pipeline.view_loc     = glGetUniformLocation(pipeline.prog.get(), "view");
        pipeline.proj_loc     = glGetUniformLocation(pipeline.prog.get(), "proj");
        pipeline.norm_loc     = glGetUniformLocation(pipeline.prog.get(), "norm");
        pipeline.rot_loc      = glGetUniformLocation(pipeline.prog.get(), "rot");


        // Setup vbo
        pipeline.center_buffer.reset(gl::create_buffer());
        pipeline.offset_buffer.reset(gl::create_buffer());
        pipeline.normal_buffer.reset(gl::create_buffer());
        pipeline.albedo_buffer.reset(gl::create_buffer());
        pipeline.texcoord_buffer.reset(gl::create_buffer());
        pipeline.index_buffer.reset(gl::create_buffer());

        // Initialize buffers
        glBindBuffer(GL_ARRAY_BUFFER, pipeline.center_buffer.get());
        glBufferData(GL_ARRAY_BUFFER,
                     centers.size() * sizeof(vec3),
                     centers.data(),
                     GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, pipeline.offset_buffer.get());
        glBufferData(GL_ARRAY_BUFFER,
                     offsets.size() * sizeof(vec3),
                     offsets.data(),
                     GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, pipeline.normal_buffer.get());
        glBufferData(GL_ARRAY_BUFFER,
                     normals.size() * sizeof(vec3),
                     normals.data(),
                     GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, pipeline.albedo_buffer.get());
        glBufferData(GL_ARRAY_BUFFER,
                     albedo.size() * sizeof(vec3),
                     albedo.data(),
                     GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, pipeline.texcoord_buffer.get());
        glBufferData(GL_ARRAY_BUFFER,
                     texcoords.size() * sizeof(vec2),
                     texcoords.data(),
                     GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pipeline.index_buffer.get());
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     indices.size() * sizeof(unsigned),
                     indices.data(),
                     GL_STATIC_DRAW);

        // Initialize logo texture
        png_image png;
        png.load(std::string(APPDIR) + "/OpenGL_White_500px_June16.png");

        pipeline.logo.reset(gl::create_texture());
        glBindTexture(GL_TEXTURE_2D, pipeline.logo.get());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, png.width(), png.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, png.data());

        // Restore OpenGL state
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_array_buffer_binding);
        glBindBuffer(GL_ARRAY_BUFFER, array_buffer_binding);

        return true;
    }

    void render_gl()
    {
        if (!pipeline.initialized)
        {
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_TEXTURE_2D);

            pipeline.initialized = init_gl();

            debugcb.activate();
        }

        // Uniform rotation
        using namespace std::chrono;
        auto now = high_resolution_clock::now();
        auto secs = duration_cast<milliseconds>(now.time_since_epoch()).count();
        mat4 rot = mat4::rotation(normalize(vec3(1.0f, 0.0f, 1.0f)), secs / 2000.0f);

        // Store OpenGL state
        GLint array_buffer_binding = 0;
        GLint element_array_buffer_binding = 0;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &array_buffer_binding);
        glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &element_array_buffer_binding);

        // Draw buffers

        pipeline.prog.enable();

        mat4 view = cam.get_view_matrix();
        mat4 proj = cam.get_proj_matrix();

        mat4 temp = view;
        temp(0,3) = temp(1,3) = temp(2,3) = 0;
        temp(3,3) = 1;

        mat4 norm = transpose(inverse(temp)) * rot;

        glUniformMatrix4fv(pipeline.view_loc, 1, GL_FALSE, view.data());
        glUniformMatrix4fv(pipeline.proj_loc, 1, GL_FALSE, proj.data());
        glUniformMatrix4fv(pipeline.norm_loc, 1, GL_FALSE, norm.data());
        glUniformMatrix4fv(pipeline.rot_loc, 1, GL_FALSE, rot.data());

        glBindBuffer(GL_ARRAY_BUFFER, pipeline.center_buffer.get());
        glEnableVertexAttribArray(pipeline.center_loc);
        glVertexAttribPointer(pipeline.center_loc, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, pipeline.offset_buffer.get());
        glEnableVertexAttribArray(pipeline.offset_loc);
        glVertexAttribPointer(pipeline.offset_loc, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, pipeline.normal_buffer.get());
        glEnableVertexAttribArray(pipeline.normal_loc);
        glVertexAttribPointer(pipeline.normal_loc, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, pipeline.albedo_buffer.get());
        glEnableVertexAttribArray(pipeline.albedo_loc);
        glVertexAttribPointer(pipeline.albedo_loc, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, pipeline.texcoord_buffer.get());
        glEnableVertexAttribArray(pipeline.texcoord_loc);
        glVertexAttribPointer(pipeline.texcoord_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);

        glUniform1i(pipeline.logo_loc, 0);
        glActiveTexture(GL_TEXTURE0 + 0);
        glBindTexture(GL_TEXTURE_2D, pipeline.logo.get());

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pipeline.index_buffer.get());

        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        glDisableVertexAttribArray(pipeline.texcoord_loc);
        glDisableVertexAttribArray(pipeline.normal_loc);
        glDisableVertexAttribArray(pipeline.offset_loc);
        glDisableVertexAttribArray(pipeline.center_loc);

        pipeline.prog.disable();

        // Restore OpenGL state
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_array_buffer_binding);
        glBindBuffer(GL_ARRAY_BUFFER, array_buffer_binding);
    }

protected:

    void on_display();
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_space_mouse_move(visionaray::space_mouse_event const& event);
    void on_resize(int w, int h);

};


//-------------------------------------------------------------------------------------------------
// Display function, contains the rendering kernel
//

void renderer::on_display()
{
    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_FRAMEBUFFER_SRGB);

    // Execute the OpenGL rendering pipeline that
    // will fill the depth buffer
    render_gl();


    // some setup for visionaray rendering

    pixel_sampler::basic_jittered_blend_type<float> blend_params;
    blend_params.spp = spp;
    float alpha = 1.0f / ++frame_num;
    blend_params.sfactor = alpha;
    blend_params.dfactor = 1.0f - alpha;

    // "Retrieve" our camera matrices; here we just retrieve
    // them from the pinhole camera that we already have. An
    // older (< OpenGL 3) application might as well do something
    // like the following:
    //
    // mat4 view;
    // mat4 proj;
    // glGetFloatv(GL_MODELVIEW_MATRIX, view.data());
    // glGetFloatv(GL_PROJECTION_MATRIX, proj.data());

    mat4 view = cam.get_view_matrix();
    mat4 proj = cam.get_proj_matrix();

    // Here we simplify this a bit and just use the view/proj
    // matrices supplied by our pinhole camera
    matrix_camera glcam(view, proj);


    auto sparams = make_sched_params(
            blend_params,
            glcam,
            device_rt
            );

    thrust::device_vector<cuda_index_bvh<basic_sphere<float>>::bvh_ref> device_primitives;
    device_primitives.push_back(device_bvh.ref());

    auto kparams = make_kernel_params(
            thrust::raw_pointer_cast(device_primitives.data()),
            thrust::raw_pointer_cast(device_primitives.data()) + device_primitives.size(),
            thrust::raw_pointer_cast(device_materials.data()),
            50,
            1E-3f,
            vec4(background_color(), 1.0f),
            vec4(0.5f, 0.7f, 1.0f, 1.0f)
            );

    pathtracing::kernel<decltype(kparams)> kern;
    kern.params = kparams;

    device_sched.frame(kern, sparams);


    // display the rendered image
    device_rt.display_color_buffer();
}


//-------------------------------------------------------------------------------------------------
// resize event
//

void renderer::on_resize(int w, int h)
{
    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    device_rt.resize(w, h);
    device_rt.clear_color_buffer();
    device_rt.clear_depth_buffer();
    frame_num = 0;

    viewer_type::on_resize(w, h);
}


//-------------------------------------------------------------------------------------------------
// mouse move event
//

void renderer::on_mouse_move(visionaray::mouse_event const& event)
{
    if (event.buttons() != mouse::NoButton)
    {
        frame_num = 0;
        device_rt.clear_color_buffer();
        device_rt.clear_depth_buffer();
    }

    viewer_type::on_mouse_move(event);
}

void renderer::on_space_mouse_move(visionaray::space_mouse_event const& event)
{
    frame_num = 0;
    device_rt.clear_color_buffer();
    device_rt.clear_depth_buffer();

    viewer_type::on_space_mouse_move(event);
}


//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

int main(int argc, char** argv)
{
    renderer rend;

    try
    {
        rend.init(argc, argv);
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend.cam.view_all( rend.bbox );

    rend.add_manipulator( std::make_shared<arcball_manipulator>(rend.cam, mouse::Left) );
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
    // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Left, keyboard::Alt) );
    rend.add_manipulator( std::make_shared<zoom_manipulator>(rend.cam, mouse::Right) );

    rend.event_loop();
}
