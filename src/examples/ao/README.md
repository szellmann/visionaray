Visionaray Ambient Occlusion Example
------------------------------------

### Command line

```
Usage:
   ao [OPTIONS] filename

Positional options:
   filename               Input file in wavefront obj format

Options:
   -bgcolor               Background color
   -bvh=<ARG>             BVH build strategy:
      =default            - Binned SAH
      =split              - Binned SAH with spatial splits
   -camera=<ARG>          Text file with camera parameters
   -fullscreen            Full screen window
   -height=<ARG>          Window height
   -radius=<ARG>          Ambient occlusion radius
   -samples=<ARG>         Number of shadow rays for ambient occlusion
   -width=<ARG>           Window width
```

### Interaction

The Visionaray Ambient Occlusion Example supports the following mouse interaction modes and keyboard shortcuts:

* **LMB**: Rotate the scene.
* **MMB**: Pan the scene (Mac OS X: **LMB** + **Key-ALT**).
* **RMB**: Zoom into the scene.
* **Key-F5**: Toggle **full screen** mode.
* **Key-ESC**: Exit **full screen** mode.
* **Key-q**: Quit example application.
* **Key-u**: **Store** the current **camera** in the working directory (visionaray-camera.txt). Be **careful**, **old** cameras are **overwritten**.
* **Key-v**: **Load** the file "visionaray-camera.txt" from the current working directory, if it exists, and adjust the **camera** accordingly.
