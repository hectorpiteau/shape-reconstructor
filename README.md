# Shape Reconstruction Experiment Software  

![](screen1.png?raw=true)

Viewer3D, Volumetric renderer, Differential rendering, tool to experiment with 3D and shapes.

Target: DRTMCS (Digital Real-Time Motion Capture System)

## Build

```sh
export CC=/usr/bin/gcc-8
export CXX=/usr/bin/g++-8
export CUDAHOSTCXX=/usr/bin/g++-8
```

- `mkdir build`
- `cd build`
- `cmake ..`
- `make -j`


Requirements:
- libxrandr
- libxinerama
- libxcursor
- libxi
- CUDA
- GCC / G++
- Zlib