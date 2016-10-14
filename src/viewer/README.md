Visionaray Viewer
-----------------

### Command line

```
Usage:
   vsnray-viewer [OPTIONS] filename

Positional options:
   filename               Input file in wavefront obj format

Options:
   -algorithm=<ARG>       Rendering algorithm:
      =simple             - Simple ray casting kernel
      =whitted            - Whitted style ray tracing kernel
      =pathtracing        - Pathtracing global illumination kernel
   -bgcolor               Background color
   -camera=<ARG>          Text file with camera parameters
   -fullscreen            Full screen window
   -height=<ARG>          Window height
   -ssaa=<ARG>            Supersampling anti-aliasing factor:
      =1                  - 1x supersampling
      =2                  - 2x supersampling
      =4                  - 4x supersampling
      =8                  - 8x supersampling
   -width=<ARG>           Window width
```

### Interaction

The viewer supports the following mouse interaction modes and keyboard shortcuts:

* **LMB**: Rotate the scene.
* **MMB**: Pan the scene (Mac OS X: **LMB** + **Key-ALT**).
* **RMB**: Zoom into the scene.
* **Key-1**: Switch to **ray casting** algorithm (default).
* **Key-2**: Switch to **ray tracing** algorithm.
* **Key-3**: Switch to **path tracing** algorithm.
* **Key-m**: **Switch** between **CPU** mode and **GPU** mode (must be [compiled with CUDA](#build-cuda)).
* **Key-h**: Toggle visibility of head up display
* **Key-s**: Toggle supersampling anti-aliasing mode. Only applies to ray casting and ray tracing algorithm (simple|whitted). Supported modes: 1x, 2x, 4x, and 8x supersampling.
* **Key-u**: **Store** the current **camera** in the working directory (visionaray-camera.txt). Be **careful**, **old** cameras are **overwritten**.
* **Key-v**: **Load** the file "visionaray-camera.txt" from the current working directory, if it exists, and adjust the **camera** accordingly.
* **Key-F5**: Toggle **full screen** mode.
* **Key-ESC**: Exit **full screen** mode.
* **Key-q**: Quit viewer application.
