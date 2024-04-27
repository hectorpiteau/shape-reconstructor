//
// Created by hepiteau on 17/04/24.
//

#ifndef DRTMCS_SVO_H
#define DRTMCS_SVO_H
#include <glm/glm.hpp>
#include "view/Lines.hpp"
#include "view/PointCloud.h"


using namespace glm;
class SVO {
private:
    vec3 origin{0, 0, 0};
    vec3 dim{1, 1, 1};
    unsigned int maxLOD{10};
    unsigned int LOD{1};

    unsigned int data_length{3*2*100};
    float lines_data[3*2*100] = {0};

    unsigned int data_length_b{3*2*100};
    float lines_data_b[3*2*100] = {0};

    Scene* m_scene;
    Lines m_lines;

    Lines m_lines_b;
    vec3 points[100] = {};
    PointCloud pcd;

    std::vector<int> child_descriptors;

public:
    SVO(Scene* scene);
    SVO(const SVO&) = delete;

    void Init();

    void Print();

    void Render();


};


#endif //DRTMCS_SVO_H
