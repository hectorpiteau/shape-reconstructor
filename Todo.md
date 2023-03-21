# Todo list

- [x] Implement a standard mouse gimbal like in any 3D software.
- [x] Add simple ui with DearImGUI.
    - Enable disable elements in the scene.  
    - Panel to plot statistics.

### Rendering

- [x] Create the Cuda Volume class with it's wireframe bouding box renderer.
- [ ] Implement the sparse octree for storing volumetric data. 
- [ ] Implement an Adam Optimizer in CUDA to optimize the volume colors. 
- [ ] Create a list of elements present in the scene and a list in ImGUI to enable / disable the render of each elements.

### Calibration

- [ ] Implement a button in ImGUI to load a set of uncalibrated images.
- [ ] Implement a panel to visualize images rapidly for manual inspection.
- [ ] Implement a SIFT extraction pipeline using OpenCV. 
- [ ] Implement a simple calibration using a regular chess grid to position images in space, (find extrinsics).
    - Add a checkbox to tell if each images have comming from the same camera or not, if same camera (algo can solve the intrinsics matrix).