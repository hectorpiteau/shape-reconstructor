//
// Created by hepiteau on 17/04/24.
//

#include "SVO.h"
#include "Projection.cuh"
#include <iostream>
#include <vector>
#include <bitset>


SVO::SVO(Scene* scene):m_scene(scene), m_lines(scene, lines_data,data_length ), m_lines_b(scene, lines_data_b,data_length_b ), pcd(scene, points, 10) {
    auto step = dim / vec3(maxLOD);
    unsigned int fi = 0;
    for(unsigned int i=0; i < maxLOD; i++ ){
        auto tmp_a = vec3(i * step.x, 0, 0.0f);
        auto tmp_b = vec3(i * step.x, 0, 1.0f);
        WRITE_VEC3(lines_data, fi, tmp_a)
        fi +=3;
        WRITE_VEC3(lines_data, fi, tmp_b)
        fi +=3;
    }

    for(unsigned int i=0; i < maxLOD; i++ ){
        auto tmp_a = vec3(0.0f, 0, i * step.z);
        auto tmp_b = vec3(1.0f, 0, i * step.z);
        WRITE_VEC3(lines_data, fi, tmp_a)
        fi +=3;
        WRITE_VEC3(lines_data, fi, tmp_b)
        fi +=3;
    }

    //Define ray
    vec3 rorigin = vec3(-0.3, 0, -0.3);
    vec3 rtarget = vec3(1.2, 0, 1.35);
    vec3 rdir = glm::normalize(rtarget - rorigin);
    WRITE_VEC3(lines_data_b, 0, rorigin)
    WRITE_VEC3(lines_data_b, 3, rtarget)
    m_lines_b.SetColor(vec4(1.0, 0.0, 0.0, 1.0));

    //(vec3 origin, vec3 dir, vec3 bbox_min, vec3 bbox_max, float* tmin, float* tmax)
    float tmin = 0.0f, tmax = 0.0f;
    vec3 bbmin = vec3(0,0,0);
    vec3 bbmax = vec3( step.x,0, step.z);

    int cpt = 0;
    for(int i=0; i<10; i++){
        BBoxTminTmax(rorigin, rdir, bbmin, bbmax, &tmin, &tmax);
        points[cpt++] = rorigin + normalize(rdir) * tmax;
        bbmax.x += step.x;
        BBoxTminTmax(rorigin, rdir, bbmin, bbmax, &tmin, &tmax);
        points[cpt++] = rorigin + normalize(rdir) * tmax;
        bbmax.z += step.z;
        BBoxTminTmax(rorigin, rdir, bbmin, bbmax, &tmin, &tmax);
        points[cpt++] = rorigin + normalize(rdir) * tmax;
        bbmax.x -= step.x;
        BBoxTminTmax(rorigin, rdir, bbmin, bbmax, &tmin, &tmax);
        points[cpt++] = rorigin + normalize(rdir) * tmax;
        bbmax.x += step.x;
    }

    m_lines.UpdateVertices(lines_data);
    m_lines_b.UpdateVertices(lines_data_b);
    pcd.UpdatePoints(points, 100);
}

void SVO::Render() {
    m_lines.Render();
    m_lines_b.Render();
    pcd.Render();
}


// Define a structure to initialize the sparse voxel octree (SVO)
void SVO::initialize_svo(std::vector<int>& child_descriptors, int depth, int current_index, int& next_free_index) {
    // Base case: if we reach the maximum depth, set the leaf mask and valid mask to indicate all leaf children
    if (depth == 0) {
        int leaf_mask = 0x000000FF; // All 8 children are leaves
        int valid_mask = 0x0000FF00; // All 8 children are valid
        int child_pointer = 0; // No further children

        int child_descriptor = (leaf_mask) | (valid_mask) | (child_pointer);
        child_descriptors[current_index] = child_descriptor;

        return;
    }

    // Otherwise, create a new child descriptor
    int child_pointer = next_free_index; // Next free index in the array
    int valid_mask = 0xFF; // All 8 children are valid
    int leaf_mask = 0x00; // No leaves at this level

    // Store the current child descriptor
    int child_descriptor = (leaf_mask << 24) | (valid_mask << 16) | (child_pointer);
    child_descriptors[current_index] = child_descriptor;

    // Now recursively initialize the children
    for (int i = 0; i < 8; ++i) {
        int child_index = child_pointer + i;
        next_free_index++; // Move to the next free index
        initialize_svo(child_descriptors, depth - 1, child_index, next_free_index);
    }
}


void SVO::Init() {
    // Allocate child desc vector
    std::vector<int> v_child_desc(1000, 0);
    // Allocate data vector
    std::vector<int> v_data(1000, 0);

    // Set the amount of child per baby block.
    int childs_per_bblock = 64; // 4x4x4 = 2 levels of divisions.
    int next_free_index = 0;


}

void SVO::Print() {
    for (size_t i = 0; i < child_descriptors.size(); ++i) {
        // Convert each integer to a 32-bit binary string
        std::bitset<32> binary_representation(child_descriptors[i]);
        // Print the index and the corresponding binary string
        std::cout << "Descriptor[" << i << "]: " << binary_representation << std::endl;
    }
}

bool BBTminTmax(const glm::vec3& origin, const glm::vec3& dir, const glm::vec3& bbox_min, const glm::vec3& bbox_max, float* tmin, float* tmax) {
    //TODO use FMA
    glm::vec3 ray_inv = 1.0f / dir;
    float tx1 = (bbox_min.x - origin.x) * ray_inv.x;
    float tx2 = (bbox_max.x - origin.x) * ray_inv.x;
    *tmin = min(tx1, tx2);
    *tmax = max(tx1, tx2);
    float ty1 = (bbox_min.y - origin.y) * ray_inv.y;
    float ty2 = (bbox_max.y - origin.y) * ray_inv.y;
    *tmin = max(*tmin, min(ty1, ty2));
    *tmax = min(*tmax, max(ty1, ty2));
    float tz1 = (bbox_min.z - origin.z) * ray_inv.z;
    float tz2 = (bbox_max.z - origin.z) * ray_inv.z;
    *tmin = max(*tmin, min(tz1, tz2));
    *tmax = min(*tmax, max(tz1, tz2));
    return *tmax >= max(0.0f, *tmin);
}


void SVO::Traverse(const glm::vec3& ray, std::vector<glm::vec3>* out_inter_pts){
    uint        currentLOD      = 0;
    int32_t     cdPtr           = 0;

    glm::vec3   SVOWorldDim     = {1.0, 1.0, 1.0};
    glm::vec3   SVOWorldOrigin  = {1.0, 1.0, 1.0};
    float       start_time,
                end_time;

    auto        res             = BBTminTmax(SVOWorldOrigin,
                                             ray,
                                             SVOWorldOrigin,
                                             SVOWorldOrigin + SVOWorldDim,
                                             &start_time,
                                             &end_time);
    // If res is false, the ray does not intersect the SVO.
    if (res == false) return;

    glm::vec3   start_pos       = ray * start_time;
    glm::vec3   end_pos         = ray * end_time;

    // Retrieve the starting child descriptor
    auto currentCD = this->child_descriptors[0];

    // Find the first child entered by the ray.




}
