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
   (tested with g++-4.8.2 on Ubuntu 14.04 x86_64,
    tested with clang++-6.0 on Mac OS X 10.10)

- [CMake][1] version 2.8 or newer
- [Boost][2]
- OpenGL
- [GLEW][3]
- [NVIDIA CUDA Toolkit][4] version 6.5 or newer (optional)

- All external dependencies but CMake are required as developer packages containing C/C++ header files
- In the future we intend to relax the OpenGL and GLEW dependency
- Microsoft Windows support is there but poor, most testing is done under Linux and Mac OS X
- When targeting NVIDIA CUDA, make sure you have a C++11 compliant version (v6.5 or later on Windows and Linux, v7.0 on Mac OS X)
- Visionaray supports Fermi+ NVIDIA GPUs (e.g. >= GeForce 400 series or >= Quadro {4|5|6}000)

Additionally, in order to compile the viewer application, the following packages are needed:

- [GLUT][5] or [FreeGLUT][6]
- [Libjpeg][7] (optional)



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
