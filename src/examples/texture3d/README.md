Visionaray 3D Texture Example
-----------------------------

### Command line

```
Usage:
   texture3d [OPTIONS] filename

Positional options:
   filename               Input file in wavefront obj format

Options:
   -bgcolor               Background color
   -color1                First color
   -color2                Second color
   -fullscreen            Full screen window
   -height=<ARG>          Window height
   -texsize               Size of the 3D texture
   -width=<ARG>           Window width
```

### Interaction

The Visionaray 3D Texture Example supports the following mouse interaction modes and keyboard shortcuts:

* **LMB**: Rotate the scene.
* **MMB**: Pan the scene (Mac OS X: **LMB** + **Key-ALT**).
* **RMB**: Zoom into the scene.
* **Key-F5**: Toggle **full screen** mode.
* **Key-ESC**: Exit **full screen** mode.
* **Key-q**: Quit example application.
* **Key-u**: **Store** the current **camera** in the working directory (visionaray-camera.txt). Be **careful**, **old** cameras are **overwritten**.
* **Key-v**: **Load** the file "visionaray-camera.txt" from the current working directory, if it exists, and adjust the **camera** accordingly.
