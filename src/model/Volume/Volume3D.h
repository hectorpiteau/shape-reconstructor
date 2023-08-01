//
// Created by hepiteau on 30/07/23.
//

#ifndef DRTMCS_VOLUME3D_H
#define DRTMCS_VOLUME3D_H
#include <glm/glm.hpp>
#include "Common.cuh"
#include "GPUData.cuh"
#include "../../view/SceneObject/SceneObject.hpp"

/**
 * Interface for a Volume3D in the software.
 *
 * Contains declaration for common methods.
 */
class Volume3D : public SceneObject {
public:
    Volume3D() : SceneObject{std::string("VOLUME3D"), SceneObjectTypes::VOLUME3D} {};
    /**
     * Get the volume's maximum resolution.
     * @return An ivec3 that correspond to the volume's size in each cartesian direction.
     */
    virtual const glm::ivec3 &GetResolution() = 0;

    /**
     * Get the volume's bounding-box minimum coordinates.
     * @return : A vec3 that correspond to the bbox minimum coordinates.
     */
    virtual const glm::vec3 &GetBboxMin() = 0;

    /**
     * Get the volume's bounding-box maximum coordinates.
     * @return : A vec3 that correspond to the bbox maximum coordinates.
     */
    virtual const glm::vec3 &GetBboxMax() = 0;

    /**
     * @brief Set the Volume Min Bounding-Box coordinates.
     *
     * @param bboxMin : The min vec3 corner of the bbox.
     */
    virtual void SetBBoxMin(const vec3 &bboxMin) = 0;

    /**
     * @brief Set the Volume Max Bounding-Box coordinates.
     *
     * @param bboxMin : The max vec3 corner of the bbox.
     */
    virtual  void SetBBoxMax(const vec3 &bboxMax) = 0;

    /**
     * Get a pointer to a memory location on gpu that contains a description of
     * the volume's bounding-box.
     *
     * @return A GPU-only-valid pointer to the volume's bounding-box descriptor.
     */
    virtual BBoxDescriptor* GetBBoxGPUDescriptor() = 0;

    /**
     * Get a pointer to a memory location on gpu that contains a description of
     * the volume's itself with data included.
     *
     * @return A GPU-only-valid pointer to the volume's data descriptor.
     */
    virtual GPUData<VolumeDescriptor>* GetGPUData() = 0;
};

#endif //DRTMCS_VOLUME3D_H
