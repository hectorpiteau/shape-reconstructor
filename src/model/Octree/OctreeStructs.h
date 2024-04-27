//
// Created by hepiteau on 23/03/24.
//

#ifndef DRTMCS_OCTREESTRUCTS_H
#define DRTMCS_OCTREESTRUCTS_H

#include <cstdint>

#define BLOCK_SIZE 1024

#define MAX_FAR_POINTER_OFFSET 32768
#define HEADER_OFFSET 8192


#define GET_FAR_BIT(x) (((x) >> 15) & 1)
#define GET_FAR_BIT_V2(x) (unsigned char)((x) & 0x01000000)
#define GET_LEAF_MASK(x) (unsigned char)((x) & 0xFF)
#define GET_VALID_MASK(x) (unsigned char)((x) & 0xFF00)
#define GET_CHILD_POINTER(x) (unsigned char)((x) & 0xFE000000)

struct line {
    uint_fast32_t ptr;

    static bool get_far_bit(const int& x) { return GET_FAR_BIT(x); }
    static unsigned char get_leaf_mask(const int& x) { return GET_LEAF_MASK(x); }
    static unsigned char get_valid_mask(const int& x) { return GET_VALID_MASK(x); }
    static unsigned char get_child_pointer(const int& x) { return GET_CHILD_POINTER(x); }
};
struct info_section {
    int* first_child_desc;
};

struct block {
    int child_desc[BLOCK_SIZE];

    struct info_section infos;
};

void CreateStubOctree(){
    struct block b1 {};
    for(int i=0; i<BLOCK_SIZE; ++i){
        b1.child_desc[i] =
    }
}

#endif //DRTMCS_OCTREESTRUCTS_H
