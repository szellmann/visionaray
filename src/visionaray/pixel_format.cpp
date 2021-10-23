// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/config.h>

#include <cassert>

#include <algorithm>
#include <utility>
#include <vector>

#if VSNRAY_HAVE_GLEW
#include <GL/glew.h>
#elif VSNRAY_HAVE_OPENGLES
#include <GLES2/gl2.h>
#endif

#include <visionaray/pixel_format.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Map GL format and size to pixel format
//

struct format_key
{
    unsigned format;
    unsigned type;
    unsigned size;

    bool operator<(format_key const& key) const
    {
        return format < key.format;
    }
};

static std::vector<std::pair<format_key, pixel_format>> gl_formats(
{
#if defined(GL_VERSION_1_1) && GL_VERSION_1_1 // TODO!
    { { GL_R8,                      GL_UNSIGNED_BYTE,                   1 },                PF_R8                   },
    { { GL_RG8,                     GL_UNSIGNED_BYTE,                   2 },                PF_RG8                  },
    { { GL_RGB8,                    GL_UNSIGNED_BYTE,                   3 },                PF_RGB8                 },
    { { GL_RGBA8,                   GL_UNSIGNED_BYTE,                   4 },                PF_RGBA8                },
    { { GL_R16F,                    GL_HALF_FLOAT,                      2 },                PF_R16F                 },
    { { GL_RG16F,                   GL_HALF_FLOAT,                      4 },                PF_RG16F                },
    { { GL_RGB16F,                  GL_HALF_FLOAT,                      6 },                PF_RGB16F               },
    { { GL_RGBA16F,                 GL_HALF_FLOAT,                      8 },                PF_RGBA16F              },
    { { GL_R32F,                    GL_FLOAT,                           4 },                PF_R32F                 },
    { { GL_RG32F,                   GL_FLOAT,                           8 },                PF_RG32F                },
    { { GL_RGB32F,                  GL_FLOAT,                          12 },                PF_RGB32F               },
    { { GL_RGBA32F,                 GL_FLOAT,                          16 },                PF_RGBA32F              },
    { { GL_R16I,                    GL_INT,                             2 },                PF_R16I                 },
    { { GL_RG16I,                   GL_INT,                             4 },                PF_RG16I                },
    { { GL_RGB16I,                  GL_INT,                             6 },                PF_RGB16I               },
    { { GL_RGBA16I,                 GL_INT,                             8 },                PF_RGBA16I              },
    { { GL_R32I,                    GL_INT,                             4 },                PF_R32I                 },
    { { GL_RG32I,                   GL_INT,                             8 },                PF_RG32I                },
    { { GL_RGB32I,                  GL_INT,                            12 },                PF_RGB32I               },
    { { GL_RGBA32I,                 GL_INT,                            16 },                PF_RGBA32I              },
    { { GL_R16UI,                   GL_UNSIGNED_INT,                    2 },                PF_R16UI                },
    { { GL_RG16UI,                  GL_UNSIGNED_INT,                    4 },                PF_RG16UI               },
    { { GL_RGB16UI,                 GL_UNSIGNED_INT,                    6 },                PF_RGB16UI              },
    { { GL_RGBA16UI,                GL_UNSIGNED_INT,                    8 },                PF_RGBA16UI             },
    { { GL_R32UI,                   GL_UNSIGNED_INT,                    4 },                PF_R32UI                },
    { { GL_RG32UI,                  GL_UNSIGNED_INT,                    8 },                PF_RG32UI               },
    { { GL_RGB32UI,                 GL_UNSIGNED_INT,                   12 },                PF_RGB32UI              },
    { { GL_RGBA32UI,                GL_UNSIGNED_INT,                   16 },                PF_RGBA32UI             },


    { { GL_RED,                     GL_UNSIGNED_BYTE,                   1 },                PF_R8                   },
    { { GL_RG,                      GL_UNSIGNED_BYTE,                   2 },                PF_RG8                  },
    { { GL_RGB,                     GL_UNSIGNED_BYTE,                   3 },                PF_RGB8                 },
    { { GL_RGBA,                    GL_UNSIGNED_BYTE,                   4 },                PF_RGBA8                },
    { { GL_RED,                     GL_HALF_FLOAT,                      2 },                PF_R16F                 },
    { { GL_RG,                      GL_HALF_FLOAT,                      4 },                PF_RG16F                },
    { { GL_RGB,                     GL_HALF_FLOAT,                      6 },                PF_RGB16F               },
    { { GL_RGBA,                    GL_HALF_FLOAT,                      8 },                PF_RGBA16F              },
    { { GL_RED,                     GL_FLOAT,                           4 },                PF_R32F                 },
    { { GL_RG,                      GL_FLOAT,                           8 },                PF_RG32F                },
    { { GL_RGB,                     GL_FLOAT,                          12 },                PF_RGB32F               },
    { { GL_RGBA,                    GL_FLOAT,                          16 },                PF_RGBA32F              },
    { { GL_RED_INTEGER,             GL_INT,                             2 },                PF_R16I                 },
    { { GL_RG_INTEGER,              GL_INT,                             4 },                PF_RG16I                },
    { { GL_RGB_INTEGER,             GL_INT,                             6 },                PF_RGB16I               },
    { { GL_RGBA_INTEGER,            GL_INT,                             8 },                PF_RGBA16I              },
    { { GL_RED_INTEGER,             GL_INT,                             4 },                PF_R32I                 },
    { { GL_RG_INTEGER,              GL_INT,                             8 },                PF_RG32I                },
    { { GL_RGB_INTEGER,             GL_INT,                            12 },                PF_RGB32I               },
    { { GL_RGBA_INTEGER,            GL_INT,                            16 },                PF_RGBA32I              },
    { { GL_RED_INTEGER,             GL_UNSIGNED_INT,                    2 },                PF_R16UI                },
    { { GL_RG_INTEGER,              GL_UNSIGNED_INT,                    4 },                PF_RG16UI               },
    { { GL_RGB_INTEGER,             GL_UNSIGNED_INT,                    6 },                PF_RGB16UI              },
    { { GL_RGBA_INTEGER,            GL_UNSIGNED_INT,                    8 },                PF_RGBA16UI             },
    { { GL_RED_INTEGER,             GL_UNSIGNED_INT,                    4 },                PF_R32UI                },
    { { GL_RG_INTEGER,              GL_UNSIGNED_INT,                    8 },                PF_RG32UI               },
    { { GL_RGB_INTEGER,             GL_UNSIGNED_INT,                   12 },                PF_RGB32UI              },
    { { GL_RGBA_INTEGER,            GL_UNSIGNED_INT,                   16 },                PF_RGBA32UI             },

    { { GL_BGR,                     GL_UNSIGNED_BYTE,                   3 },                PF_BGR8                 },
    { { GL_BGRA,                    GL_UNSIGNED_BYTE,                   4 },                PF_BGRA8                },

    { { GL_RGB10_A2,                GL_UNSIGNED_INT_10_10_10_2,         4 },                PF_RGB10_A2             },
    { { GL_R11F_G11F_B10F,          GL_UNSIGNED_INT_10F_11F_11F_REV,    3 },                PF_R11F_G11F_B10F       },
    { { GL_RGBA,                    GL_UNSIGNED_INT_10_10_10_2,         4 },                PF_RGB10_A2             },
    { { GL_RGB,                     GL_UNSIGNED_INT_10F_11F_11F_REV,    3 },                PF_R11F_G11F_B10F       },

    { { GL_DEPTH_COMPONENT16,       GL_UNSIGNED_INT,                    2 },                PF_DEPTH16              },
    { { GL_DEPTH_COMPONENT24,       GL_UNSIGNED_INT,                    3 },                PF_DEPTH24              },
    { { GL_DEPTH_COMPONENT32,       GL_UNSIGNED_INT,                    4 },                PF_DEPTH32              },
    { { GL_DEPTH_COMPONENT32F,      GL_FLOAT,                           4 },                PF_DEPTH32F             },
    { { GL_DEPTH24_STENCIL8,        GL_UNSIGNED_INT_24_8,               4 },                PF_DEPTH24_STENCIL8     },
    { { GL_DEPTH32F_STENCIL8,       GL_FLOAT_32_UNSIGNED_INT_24_8_REV,  8 },                PF_DEPTH32F_STENCIL8    },

    { { GL_DEPTH_COMPONENT,         GL_UNSIGNED_INT,                    2 },                PF_DEPTH16              },
    { { GL_DEPTH_COMPONENT,         GL_UNSIGNED_INT,                    3 },                PF_DEPTH24              },
    { { GL_DEPTH_COMPONENT,         GL_UNSIGNED_INT,                    4 },                PF_DEPTH32              },
    { { GL_DEPTH_COMPONENT,         GL_FLOAT,                           4 },                PF_DEPTH32F             },
    { { GL_DEPTH_STENCIL,           GL_UNSIGNED_INT_24_8,               4 },                PF_DEPTH24_STENCIL8     },
    { { GL_DEPTH_STENCIL,           GL_FLOAT_32_UNSIGNED_INT_24_8_REV,  8 },                PF_DEPTH32F_STENCIL8    },

    { { GL_LUMINANCE8,              GL_UNSIGNED_BYTE,                   1 },                PF_LUMINANCE8           },
    { { GL_LUMINANCE16,             GL_UNSIGNED_SHORT,                  2 },                PF_LUMINANCE16          },
    { { GL_LUMINANCE32F_ARB,        GL_FLOAT,                           4 },                PF_LUMINANCE32F         },

    { { GL_LUMINANCE,               GL_UNSIGNED_BYTE,                   1 },                PF_LUMINANCE8           },
    { { GL_LUMINANCE,               GL_UNSIGNED_SHORT,                  2 },                PF_LUMINANCE16          },
    { { GL_LUMINANCE,               GL_FLOAT,                           4 },                PF_LUMINANCE32F         },
#elif defined(GL_ES_VERSION_2_0) && GL_ES_VERSION_2_0
    { { GL_RGB,                     GL_UNSIGNED_BYTE,                   3 },                PF_RGB8                 },
    { { GL_RGBA,                    GL_UNSIGNED_BYTE,                   4 },                PF_RGBA8                },
    { { GL_LUMINANCE,               GL_UNSIGNED_BYTE,                   1 },                PF_LUMINANCE8           },
#endif
});

pixel_format map_gl_format(unsigned format, unsigned type, unsigned size)
{
    format_key key;
    key.format = format;
    key.type   = type;
    key.size   = size;

    auto pf = std::find_if(
            gl_formats.begin(),
            gl_formats.end(),
            [&](std::pair<format_key, pixel_format> const& value) -> bool
            {
                auto val = value.first;
                return val.format == key.format && val.type == key.type && val.size == key.size;
            }
            );

    if (pf == gl_formats.end())
    {
        return PF_UNSPECIFIED;
    }

    return pf->second;
}


//-------------------------------------------------------------------------------------------------
// Map pixel formats to pixel format infos
//

static const pixel_format_info color_formats[] =
{

    { 0, 0, 0, 0, 0 }, // PF_UNSPECIFIED

#if defined(GL_VERSION_1_1) && GL_VERSION_1_1 // TODO!

    //----------------------------------------------------------------------------------------------
    // for colors etc.
    //

    { GL_R8,                    GL_RED,                 GL_UNSIGNED_BYTE,                   1,  1   },      // PF_R8
    { GL_RG8,                   GL_RG,                  GL_UNSIGNED_BYTE,                   2,  2   },      // PF_RG8
    { GL_RGB8,                  GL_RGB,                 GL_UNSIGNED_BYTE,                   3,  3   },      // PF_RGB8
    { GL_RGBA8,                 GL_RGBA,                GL_UNSIGNED_BYTE,                   4,  4   },      // PF_RGBA8
    { GL_R16F,                  GL_RED,                 GL_HALF_FLOAT,                      1,  2   },      // PF_R16F
    { GL_RG16F,                 GL_RG,                  GL_HALF_FLOAT,                      2,  4   },      // PF_RG16F
    { GL_RGB16F,                GL_RGB,                 GL_HALF_FLOAT,                      3,  6   },      // PF_RGB16F
    { GL_RGBA16F,               GL_RGBA,                GL_HALF_FLOAT,                      4,  8   },      // PF_RGBA16F
    { GL_R32F,                  GL_RED,                 GL_FLOAT,                           1,  4   },      // PF_R32F
    { GL_RG32F,                 GL_RG,                  GL_FLOAT,                           2,  8   },      // PF_RG32F
    { GL_RGB32F,                GL_RGB,                 GL_FLOAT,                           3, 12   },      // PF_RGB32F
    { GL_RGBA32F,               GL_RGBA,                GL_FLOAT,                           4, 16   },      // PF_RGBA32F
    { GL_R16I,                  GL_RED_INTEGER,         GL_INT,                             1,  2   },      // PF_R16I
    { GL_RG16I,                 GL_RG_INTEGER,          GL_INT,                             2,  4   },      // PF_RG16I
    { GL_RGB16I,                GL_RGB_INTEGER,         GL_INT,                             3,  6   },      // PF_RGB16I
    { GL_RGBA16I,               GL_RGBA_INTEGER,        GL_INT,                             4,  8   },      // PF_RGBA16I
    { GL_R32I,                  GL_RED_INTEGER,         GL_INT,                             1,  4   },      // PF_R32I
    { GL_RG32I,                 GL_RG_INTEGER,          GL_INT,                             2,  8   },      // PF_RG32I
    { GL_RGB32I,                GL_RGB_INTEGER,         GL_INT,                             3, 12   },      // PF_RGB32I
    { GL_RGBA32I,               GL_RGBA_INTEGER,        GL_INT,                             4, 16   },      // PF_RGBA32I
    { GL_R16UI,                 GL_RED_INTEGER,         GL_UNSIGNED_INT,                    1,  2   },      // PF_R16UI
    { GL_RG16UI,                GL_RG_INTEGER,          GL_UNSIGNED_INT,                    2,  4   },      // PF_RG16UI
    { GL_RGB16UI,               GL_RGB_INTEGER,         GL_UNSIGNED_INT,                    3,  6   },      // PF_RGB16UI
    { GL_RGBA16UI,              GL_RGBA_INTEGER,        GL_UNSIGNED_INT,                    4,  8   },      // PF_RGBA16UI
    { GL_R32UI,                 GL_RED_INTEGER,         GL_UNSIGNED_INT,                    1,  4   },      // PF_R32UI,
    { GL_RG32UI,                GL_RG_INTEGER,          GL_UNSIGNED_INT,                    2,  8   },      // PF_RG32UI,
    { GL_RGB32UI,               GL_RGB_INTEGER,         GL_UNSIGNED_INT,                    3, 12   },      // PF_RGB32UI,
    { GL_RGBA32UI,              GL_RGBA_INTEGER,        GL_UNSIGNED_INT,                    4, 16   },      // PF_RGBA32UI,

    { GL_RGB8,                  GL_BGR,                 GL_UNSIGNED_BYTE,                   3,  3   },      // PF_BGR8
    { GL_RGBA8,                 GL_BGRA,                GL_UNSIGNED_BYTE,                   4,  4   },      // PF_BGRA8

    { GL_RGB10_A2,              GL_RGBA,                GL_UNSIGNED_INT_10_10_10_2,         4,  4   },      // PF_RGB10_A2
    { GL_R11F_G11F_B10F,        GL_RGB,                 GL_UNSIGNED_INT_10F_11F_11F_REV,    3,  4   },      // PF_R11F_G11F_B10F

    //----------------------------------------------------------------------------------------------
    // for depth / stencil buffers
    //

    { GL_DEPTH_COMPONENT16,     GL_DEPTH_COMPONENT,     GL_UNSIGNED_INT,                    1,  2   },      // PF_DEPTH16
    { GL_DEPTH_COMPONENT24,     GL_DEPTH_COMPONENT,     GL_UNSIGNED_INT,                    1,  3   },      // PF_DEPTH24
    { GL_DEPTH_COMPONENT32,     GL_DEPTH_COMPONENT,     GL_UNSIGNED_INT,                    1,  4   },      // PF_DEPTH32
    { GL_DEPTH_COMPONENT32F,    GL_DEPTH_COMPONENT,     GL_FLOAT,                           1,  4   },      // PF_DEPTH32F
    { GL_DEPTH24_STENCIL8,      GL_DEPTH_STENCIL,       GL_UNSIGNED_INT_24_8,               2,  4   },      // PF_DEPTH24_STENCIL8
    { GL_DEPTH32F_STENCIL8,     GL_DEPTH_STENCIL,       GL_FLOAT_32_UNSIGNED_INT_24_8_REV,  2,  8   },      // PF_DEPTH32F_STENCIL8

    { GL_LUMINANCE8,            GL_LUMINANCE,           GL_UNSIGNED_BYTE,                   1,  1   },      // PF_LUMINANCE8
    { GL_LUMINANCE16,           GL_LUMINANCE,           GL_UNSIGNED_SHORT,                  1,  2   },      // PF_LUMINANCE16
    { GL_LUMINANCE32F_ARB,      GL_LUMINANCE,           GL_FLOAT,                           1,  4   }       // PF_LUMINANCE32F

#elif defined(GL_ES_VERSION_2_0) && GL_ES_VERSION_2_0

    { GL_RGB,                   GL_RGB,                 GL_UNSIGNED_BYTE,                   3,  3   },      // PF_RGB8
    { GL_RGBA,                  GL_RGBA,                GL_UNSIGNED_BYTE,                   4,  4   },      // PF_RGBA8
    { GL_LUMINANCE,             GL_LUMINANCE,           GL_UNSIGNED_BYTE,                   1,  1   },      // PF_LUMINANCE8

#endif
};


pixel_format_info map_pixel_format(pixel_format format)
{
    unsigned idx = static_cast<unsigned>(format);

    assert(idx < PF_COUNT);

    return color_formats[idx];
}

} // visionaray
