#include "DenseFloat32Volume.hpp"

#include <glm/glm.hpp>
#include <limits>
#include <algorithm>

DenseFloat32Volume::DenseFloat32Volume(int resolution) : m_resolution(resolution), m_2Divresolution(2.0f/(float)resolution){
    m_data = new float[resolution*resolution*resolution];
}



DenseFloat32Volume::~DenseFloat32Volume(){
    delete [] m_data;
}


float DenseFloat32Volume::GetValue(const glm::vec3& pos){
    /** Convert position between (-1, -1, -1) and (1, 1, 1) to the value stored in the Volume.*/
    if(glm::any(glm::lessThan(pos, glm::vec3(-1, -1, -1))) 
    || glm::any(glm::greaterThan(pos, glm::vec3(1, 1, 1)))){
        return std::numeric_limits<float>::max();
    }

    glm::vec3 tmp = (pos + glm::vec3(1.0, 1.0, 1.0)) * (float)m_resolution / 2.0f;

    int x = glm::clamp((int)floor(tmp.x), 0, m_resolution);
    int y = glm::clamp((int)floor(tmp.y), 0, m_resolution);
    int z = glm::clamp((int)floor(tmp.z), 0, m_resolution);
    
    return m_data[x * m_resolution * m_resolution + y * m_resolution + z];
}

void DenseFloat32Volume::SetValue(const glm::vec3& pos, float value){
    /** Convert position between (-1, -1, -1) and (1, 1, 1) to the value stored in the Volume.*/
    if(glm::any(glm::lessThan(pos, glm::vec3(-1, -1, -1))) 
    || glm::any(glm::greaterThan(pos, glm::vec3(1, 1, 1)))){
        return;
    }

    glm::vec3 tmp = (pos + glm::vec3(1.0, 1.0, 1.0)) * (float)m_resolution / 2.0f;

    int x = glm::clamp((int)floor(tmp.x), 0, m_resolution);
    int y = glm::clamp((int)floor(tmp.y), 0, m_resolution);
    int z = glm::clamp((int)floor(tmp.z), 0, m_resolution);
    
    m_data[x * m_resolution * m_resolution + y * m_resolution + z] = value;
}

void DenseFloat32Volume::SetValue(int x, int y, int z, float value){
     m_data[x * m_resolution * m_resolution + y * m_resolution + z] = value;
}

int DenseFloat32Volume::GetResolution(){
    return m_resolution;
}