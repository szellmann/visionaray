[![Build Status](https://travis-ci.org/szellmann/visionaray.svg?branch=master)](https://travis-ci.org/szellmann/visionaray)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/szellmann/visionaray?svg=true&branch=master)](https://ci.appveyor.com/project/szellmann/visionaray/branch/master)
[![Join the chat at https://gitter.im/visionaray/Lobby](https://badges.gitter.im/visionaray/Lobby.svg)](https://gitter.im/visionaray/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Visionaray
==========

A C++ based, cross platform ray tracing library

> **Note that the current version of Visionaray is an early preview. At this stage, the framework, including the API, are likely to undergo frequent changes.**

Getting Visionaray
------------------

Under Linux or Mac OS X, use the following commands to locally clone Visionaray

```Shell
git clone https://github.com/szellmann/visionaray.git
cd visionaray
git submodule update --init --recursive
```

Build requirements
------------------

- C++11 compliant compiler
   (tested with g++-4.8.4 on Ubuntu 14.04 x86_64,
    tested with clang++-7.0.2 on Mac OS X 10.10,
    tested with Microsoft Visual Studio 2015 VC14 for x64)

- [CMake][1] version 2.8 or newer
- [Boost][2]
- OpenGL
- [GLEW][3]
- [NVIDIA CUDA Toolkit][4] version 7.0 or newer (optional)

- All external dependencies but CMake are required as developer packages containing C/C++ header files
- In the future we intend to relax the OpenGL and GLEW dependency
- When targeting NVIDIA CUDA, make sure you have a C++11 compliant version (v7.0 or newer)
- Visionaray supports Fermi+ NVIDIA GPUs (e.g. >= GeForce 400 series or >= Quadro {4|5|6}000)

Additionally, in order to compile the viewer application and the [examples](https://github.com/szellmann/visionaray/tree/master/src/examples), the following packages are needed:

- [GLUT][5] or [FreeGLUT][6]
- [Libjpeg][7] (optional)
- [Libpng][8] (optional)
- [LibTIFF][9] (optional)



Building the Visionaray library and viewer application
------------------------------------------------------

### Linux and Mac OS X

It is strongly recommended that you do a "release build" because otherwise the CPU code path will be "sluggish".
It is also recommended to supply an architecture flag that corresponds to the CPU architecture you are targeting.
Please ensure that you have C++11 support activated.

```Shell
cd visionaray
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-std=c++11 -msse4.1"
make
make install
```

The headers, libraries and viewer application will then be located in the standard install path of your operating system (usually /usr/local).

See the [Getting Started Guide](https://github.com/szellmann/visionaray/wiki/Getting-started) and the [Troubleshooting section](https://github.com/szellmann/visionaray/wiki/Troubleshooting) in the [Wiki](https://github.com/szellmann/visionaray/wiki) for further information.


Visionaray Viewer
-----------------

Visionaray comes with a viewer application that can process wavefront obj files containing polygons. The viewer application is primarily meant as a developer tool for debugging and testing.
After being installed, the viewer application executable can be called using the following command:

```Shell
vsnray-viewer <filename.obj>
```

where filename.obj is a wavefront obj file.

Documentation
-------------

Thorough documentation can be found in the [Wiki](https://github.com/szellmann/visionaray/wiki).

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
