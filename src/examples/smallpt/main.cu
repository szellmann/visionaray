// This file is distributed under the MIT license.
// See the LICENSE file for details.

// We need a main.cu to get cmake and CUDA working correctly.
// We also don't prefer symlinks for compatibility w/ Win32.
// Depending on your application and your environment, it may
// well be possible that you can simply compile main.cpp with
// nvcc.
#include "main.cpp"
