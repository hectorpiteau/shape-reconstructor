#ifndef VOLUME_3D
#define VOLUME_3D

#include <memory>
#include <iostream>
#include <glm/glm.hpp>

#include "../view/Renderable/Renderable.hpp"
#include "../view/Wireframe/Wireframe.hpp"
#include "../view/Lines.hpp"
#include "../view/SceneObject/SceneObject.hpp"

#include "../controllers/Scene/Scene.hpp"

#include "../cuda/CudaLinearVolume3D.cuh"

#include "CudaBuffer.hpp"

using namespace glm;
class Lines;

/**
 * @brief This class allows for the render of 3D volumetric data.
 * It is physically contained inside the unit-cube (size 1 in all directions).
 *
 * The data and processing for ray marching is dispatched to a CUDA kernel.
 *
 * This class performs the rendering of the result of the CUDA kernel in a plane that overlay the view.
 */
class Volume3D : public SceneObject
{
public:
    Volume3D(Scene *scene, ivec3 res);

    void UpdateGPUData();

    /**
     * @brief Set the Volume Min Bounding-Box coordinates.
     * 
     * @param bboxMin : The min vec3 corner of the bbox.
     */
    void SetBBoxMin(const vec3 &bboxMin);
    
    /**
     * @brief Set the Volume Max Bounding-Box coordinates.
     * 
     * @param bboxMin : The max vec3 corner of the bbox.
     */
    void SetBBoxMax(const vec3 &bboxMax);

    /**
     * @brief 
     * 
     */
    void InitializeVolume();

    void Render() override;

    void ComputeBBoxPoints();

    const ivec3 &GetResolution();

    const vec3 &GetBboxMin();
    const vec3 &GetBboxMax();

    std::shared_ptr<CudaLinearVolume3D> GetCudaVolume();

    vec3 m_bboxPoints[8] = {};

    BBoxDescriptor* GetGPUDescriptor();
    GPUData<VolumeDescriptor>& GetGPUData();
private:
    /** ext dep*/
    Scene *m_scene;

    /** in dep */
    std::shared_ptr<CudaLinearVolume3D> m_cudaVolume;
    std::shared_ptr<Lines> m_lines;

    /** bbox coordinates. */
    vec3 m_bboxMin = vec3(-1.0, -1.0, -1.5);
    vec3 m_bboxMax = vec3(1.0, 1.0, 1.5);

    ivec3 m_res;
    float m_voxelSize = 0.01f;

    GPUData<BBoxDescriptor> m_desc;
    GPUData<VolumeDescriptor> m_volumeDescriptor;


    /** wireframe coordinates. */
    float m_wireframeVertices[12 * 2 * 3] = {
        -0.5f,
        -0.5f,
        -0.5f, // first bottom line x-dir
        0.5f,
        -0.5f,
        -0.5f,

        -0.5f,
        -0.5f,
        0.5f, // second bottom line x-dir
        0.5f,
        -0.5f,
        0.5f,

        -0.5f,
        0.5f,
        -0.5f, // first top line x-dir
        0.5f,
        0.5f,
        -0.5f,

        -0.5f,
        0.5f,
        0.5f, // second top line x-dir
        0.5f,
        0.5f,
        0.5f,

        -0.5f,
        0.5f,
        -0.5f, // first back line y-dir
        -0.5f,
        -0.5f,
        -0.5f,

        -0.5f,
        0.5f,
        0.5f, // second back line y-dir
        -0.5f,
        -0.5f,
        0.5f,

        0.5f,
        0.5f,
        -0.5f, // first front line y-dir
        0.5f,
        -0.5f,
        -0.5f,

        0.5f,
        0.5f,
        0.5f, // second front line y-dir
        0.5f,
        -0.5f,
        0.5f,

        -0.5f,
        -0.5f,
        -0.5f, // first bottom line z-dir
        -0.5f,
        -0.5f,
        0.5f,

        0.5f,
        -0.5f,
        -0.5f, // second bottom line z-dir
        0.5f,
        -0.5f,
        0.5f,

        -0.5f,
        0.5f,
        -0.5f, // first top line z-dir
        -0.5f,
        0.5f,
        0.5f,

        0.5f,
        0.5f,
        -0.5f, // second top line z-dir
        0.5f,
        0.5f,
        0.5f,
    };
};

#endif // VOLUME_3D