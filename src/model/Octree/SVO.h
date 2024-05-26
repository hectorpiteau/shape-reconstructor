//
// Created by hepiteau on 17/04/24.
//

#ifndef DRTMCS_SVO_H
#define DRTMCS_SVO_H
#include "glm/glm/glm.hpp"
#include "../../view/Lines.hpp"
#include "../../view/PointCloud.h"
#include <bitset>

#define GET_FAR_BIT(x) (((x) >> 15) & 1)
#define GET_FAR_BIT_V2(x) (unsigned char)((x) & 0x01000000)
#define GET_LEAF_MASK(x) (unsigned char)((x) & 0xFF)
#define GET_VALID_MASK(x) (unsigned char)((x) & 0xFF00)
#define GET_CHILD_POINTER(x) (unsigned char)((x) & 0xFE000000)

#define PRINT_BITS_INT32(x) std::bitset<32>((x))

#define LEAF_MASK_TO_MORTON_INDEX_2x2x2(b) \
    ((b & 0x80) ? 0 : \
     (b & 0x40) ? 4 : \
     (b & 0x20) ? 2 : \
     (b & 0x10) ? 6 : \
     (b & 0x08) ? 1 : \
     (b & 0x04) ? 5 : \
     (b & 0x02) ? 3 : \
     (b & 0x01) ? 7 : -1)

#define IS_CHILD_POINTER_TOO_LARGE(x) ((x) > 0x00007FFF ? 1 : 0)

void initialize_svo(int32_t* nodes, size_t length, int current_depth, int target_depth, int& current_index) {
    if (current_depth > target_depth) return; // Base case for recursion
    if (target_depth == 0) return; // SVO of depth 0 does not exist.

    if(current_depth == 1){

    }

    int base_index = current_index;
    int children_start_index = current_index + 1;
    int children_count = 8;

    // Check if current index + number of children exceeds 15-bit storage capability
    if (children_start_index + children_count >= std::pow(2, 15)) {
        exit(1);
        // Store the far pointer in a separate array
//        far_pointers.push_back(children_start_index);
//        nodes[base_index] = (1 << 15) | static_cast<int32_t>(far_pointers.size() - 1); // set far bit and index to far pointers array
    }

    // Update current index for the next node
//    current_index += children_count; // Reserve space for 8 children
//    if (current_depth == max_depth) return; // If at max depth, do not initialize further children

    // Initialize children in Morton order
//    for (int i = 0; i < children_count; ++i) {
//        unsigned int x = (i & 4) >> 2;
//        unsigned int y = (i & 2) >> 1;
//        unsigned int z = i & 1;
//        int child_index = to_morton(x, y, z) + children_start_index;
//        initialize_svo(nodes, far_pointers, current_depth + 1, max_depth, child_index);
//    }
}

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
