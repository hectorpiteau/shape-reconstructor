/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRenderer.hpp (c) 2023
Desc: description
Created:  2023-04-14T09:48:58.410Z
Modified: 2023-04-14T09:49:57.434Z
*/


class VolumeRenderer {
public:
    VolumeRenderer(RayCaster* rayCaster);
    VolumeRenderer(const VolumeRenderer&) = delete;
    ~VolumeRenderer();

private:
    RayCaster* rayCaster;

};