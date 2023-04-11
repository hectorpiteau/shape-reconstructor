#pragma once

#include <glm/glm.hpp>


/**
 * @brief Represent a unit cube of a specific resolution.
 * Centered on (0,0,0) 
 * Minimum: (-1, -1, -1)
 * Maximum (1, 1, 1)
 */
class DenseFloat32Volume {
public:
    /**
     * @brief Construct a new Dense Float32 Volume
     * 
     * @param resolution : The amount of voxels in each directions.
     */
    DenseFloat32Volume(int resolution);
    ~DenseFloat32Volume();

    DenseFloat32Volume(const DenseFloat32Volume&) = delete;

    float GetValue(const glm::vec3& pos);

    void SetValue(const glm::vec3& pos, float value);
    void SetValue(int x, int y, int z, float value);

    int GetResolution();

private:
    float* m_data;
    int m_resolution;
    float m_2Divresolution;
};