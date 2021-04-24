[![Build Status](https://ci.appveyor.com/api/projects/status/github/szellmann/visionaray?svg=true&branch=master)](https://ci.appveyor.com/project/szellmann/visionaray/branch/master)
[![Join the chat at https://gitter.im/visionaray/Lobby](https://badges.gitter.im/visionaray/Lobby.svg)](https://gitter.im/visionaray/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Visionaray
==========

A C++ based, cross platform ray tracing library

Getting Visionaray
------------------

The Visionaray git repository can be cloned using the following commands:

```Shell
git clone --recursive https://github.com/szellmann/visionaray.git
```

An existing working copy can be updated using the following commands:

```Shell
git pull
git submodule sync
git submodule update --init --recursive
```

Build requirements
------------------

- C++11 compliant compiler
   (tested with g++-7.4.0 on Ubuntu 18.04 x86_64,
    tested with clang-900.0.39.2 on Mac OS X 10.13,
    tested with Microsoft Visual Studio 2015 VC14 for x64)

- [CMake][1] version 3.1.3 or newer
- [OpenGL][12]
- [GLEW][3]
- [NVIDIA CUDA Toolkit][4] version 7.0 or newer (optional)

- All external dependencies but CMake should be installed as developer packages containing C/C++ header files
- The OpenGL and GLEW dependency can optionally be relaxed by setting `VSNRAY_GRAPHICS_API=None` with CMake
- When targeting NVIDIA CUDA, make sure you have a C++11 compliant version (v7.0 or newer)
- Visionaray supports Fermi+ NVIDIA GPUs (e.g. >= GeForce 400 series or >= Quadro {4|5|6}000) (Visionaray does not support texturing on Fermi GPUs, bindless texture support is available with Kepler+ GPUs)

Additionally, in order to compile the viewer application and the [examples](/src/examples), the following packages are needed or recommended:

- [Boost][2]
- [GLUT][5] or [FreeGLUT][6]
- [Libjpeg][7] (optional)
- [Libpng][8] (optional)
- [LibTIFF][9] (optional)
- [OpenEXR][10] (optional)
- [Ptex][13] (optional)



Building the Visionaray library and viewer application
------------------------------------------------------

### Linux and Mac OS X

It is strongly recommended to build Visionaray in release mode, as the source code relies heavily on function inlining by the compiler, and executables may be extremely slow without that optimization.
It is also recommended to supply an architecture flag that corresponds to the CPU architecture you are targeting.

```Shell
cd visionaray
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
make
make install
```

The headers, libraries and viewer application will then be located in the standard install path of your operating system (usually `/usr/local`).

See the [Getting Started Guide](https://github.com/szellmann/visionaray/wiki/Getting-started) and the [Troubleshooting section](https://github.com/szellmann/visionaray/wiki/Troubleshooting) in the [Wiki](https://github.com/szellmann/visionaray/wiki) for further information.


Visionaray Viewer
-----------------

Visionaray comes with a viewer application that supports a number of different 3D file formats. The viewer application is primarily targeted at developers, as a tool for debugging and testing.
After being installed, the viewer application executable can be called using the following command:

```Shell
vsnray-viewer <file>
```

where `file` is either a path to a wavefront `.obj` file, a `.ply` file, or a `.pbrt` file.

Documentation
-------------

Documentation can be found in the [Wiki](https://github.com/szellmann/visionaray/wiki).


Source Code Organization
------------------------

### Library

Visionaray is a template library, so that most algorithms are implemented in header files located under `include/visionaray`.

- [include/visionaray/math](/include/visionaray/math): GLSL-inspired math templates, wrappers for SIMD types, geometric primitives
- [include/visionaray/texture](/include/visionaray/texture): texture management templates and texture access routines
- [include/visionaray](/include/visionaray): misc. ray tracing templates, BVHs, render targets, etc.

Visionaray can optionally interoperate with graphics and GPGPU APIs. Interoperability with the respective libraries is compiled into the Visionaray library. When GPU interoperability isn't requierd, chances are high that you don't need to link with Visionaray but can rather use it as a header only library.

- [include/visionaray/cuda/](/include/visionaray/cuda/), [src/visionaray/cuda](/src/visionaray/cuda): CUDA interoperability classes
- [include/visionaray/gl](/include/visionaray/gl), [src/visionaray/gl](/src/visionaray/gl): OpenGL(ES) interoperability classes

Files in `./detail` subfolders are not part of the public API. Code in namespace `detail` contains private implementation. Template class implementations go into files ending with `.inl`, which are included at the bottom of the public interface header file.

### Applications

Visionaray comes with a rudimentary viewer (see above) and a set of [example applications](/src/examples). Those are implemented in

- [src/viewer](/src/viewer): visionaray viewer application
- [src/examples](/src/examples): visionaray example applications

### Common library

The viewer application and the examples link with the Visionaray-common library that provides functionality such as windowing classes or mouse interaction. The Visionaray-common library is **not part of the public API** and interfaces may change between releases.

- [src/common](/src/common): private library used by the viewer and example applications

### Third-party libraries

- [src/3rdparty](/src/3rdparty): third-party code goes in here

The viewer application and the examples use the following third-party libraries (the Visionaray library can be built without these dependencies):
- [CmdLine](https://github.com/abolz/CmdLine) library to handle command line arguments in the viewer and example applications.
- [dear imgui](https://github.com/ocornut/imgui) library for GUI elements in the viewer and example applications.
- [PBRT-Parser](https://github.com/ingowald/pbrt-parser) library to load 3D models in [pbrt](https://github.com/mmp/pbrt-v3) format.
- [RapidJSON](http://rapidjson.org/) library for parsing JSON scene descriptions.
- [tinyply](https://github.com/ddiakopoulos/tinyply) library to load Stanford PLY models.

Revision History
----------------

See the file [CHANGELOG.md](/CHANGELOG.md) for updates on feature addition and removals, bug fixes and general changes.

Citation
--------

If you use Visionaray or some of its code for your scientific project, it would be nice if you cited this paper:
```
@inproceedings{zellmann:visionaray,
author = {Zellmann, Stefan and Wickeroth, Daniel and Lang, Ulrich},
title = {Visionaray: A Cross-Platform Ray Tracing Template Library},
booktitle = {2017 IEEE 10th Workshop on Software Engineering and Architectures for Realtime Interactive Systems (SEARIS)},
year = {2017},
publisher = {IEEE},
pages = {1-8},
}
```


License
-------

Visionaray is licensed under the MIT License (MIT)


[1]:    http://www.cmake.org/download/
[2]:    http://www.boost.org/users/download/
[3]:    http://glew.sourceforge.net/
[4]:    https://developer.nvidia.com/cuda-toolkit
[5]:    https://www.opengl.org/resources/libraries/glut/
[6]:    http://freeglut.sourceforge.net/index.php#download
[7]:    http://libjpeg.sourceforge.net/
[8]:    http://libpng.sourceforge.net
[9]:    http://www.libtiff.org
[10]:   http://www.openexr.com/
[12]:   https://www.opengl.org
[13]:   https://github.com/wdas/ptex
