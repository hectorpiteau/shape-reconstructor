#ifndef VOLUME_3D
#define VOLUME_3D

#include <memory>
#include <iostream>
#include "glm/glm/glm.hpp"

#include "../../view/Renderable/Renderable.hpp"
#include "../../view/Wireframe/Wireframe.hpp"
#include "../../view/Lines.hpp"
#include "../../view/SceneObject/SceneObject.hpp"

#include "../../controllers/Scene/Scene.hpp"

#include "CudaLinearVolume3D.cuh"

#include "../CudaBuffer.hpp"
#include "Volume3D.h"

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
class DenseVolume3D : public Volume3D
{
public:
    DenseVolume3D(Scene *scene, const ivec3& res);

    /** ********** SceneObject ********** */
    void UpdateGPUData();
    /** ********** ********** ********** */

    /** ********** Volume 3D ********** */
    void Render() override;
    void SetBBoxMin(const vec3 &bboxMin) override;
    void SetBBoxMax(const vec3 &bboxMin) override;
    const ivec3 &GetResolution() override;
    const vec3 &GetBboxMin() override;
    const vec3 &GetBboxMax() override;
    GPUData<VolumeDescriptor>* GetGPUData() override;
    /** ********** ********** ********** */

    /**
     * @brief Initialize the volume with zeros everywhere. Copy is done on GPU memory.
     * 
     */
    void InitializeZeros();

    /**
     * //TODO
     */
    void ComputeBBoxPoints();
    /**
     * //TODO
     * @return
     */
    std::shared_ptr<CudaLinearVolume3D> GetCudaVolume();

    vec3 m_bboxPoints[8] = {};

    /**
     * //TODO
     * @return
     */
    BBoxDescriptor* GetBBoxGPUDescriptor() override;

    /**
     * Resize the volume to the desired size.
     */
    void Resize(const ivec3& res);

    /**
     * //TODO
     */
    void DoubleResolution();

private:
    /** ext dep*/
    Scene *m_scene;

    /** in dep */
    std::shared_ptr<CudaLinearVolume3D> m_cudaVolume;
    std::shared_ptr<Lines> m_lines;

    /** bbox coordinates. */
    vec3 m_bboxMin = vec3(-1.0, -0.8, -1.5);
    vec3 m_bboxMax = vec3(1.0, 1.2, 1.5);

    ivec3 m_res;

    GPUData<BBoxDescriptor> m_desc;
    GPUData<DenseVolumeDescriptor> m_volumeDescriptor;

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
