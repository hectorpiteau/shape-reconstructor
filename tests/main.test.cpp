#include <gtest/gtest.h>
#include <libmorton/morton.h>
#include "../src/maths/Morton.h"
#include <iostream>
#include "../src/model/Octree/SVO.h"

TEST(MyClassTest, InitializationTest) {
//    std::vector<uint> gt = {0, 4, 32, 36, 2, 6, 34, 38, 16, 20, 48, 52, 18, 22, 50, 54, 1, 5, 33, 37, 3, 7, 35, 39, 17, 21, 49, 53, 19, 23, 51, 55, 8, 12, 40, 44, 10, 14, 42, 46, 24, 28, 56, 60, 26, 30, 58, 62, 9, 13, 41, 45, 11, 15, 43, 47, 25, 29, 57, 61, 27, 31, 59, 63};
    std::vector<uint> gt = {0, 4, 2, 6, 1, 5, 3, 7};
    std::vector<uint> computed = {};
    for(int x=0; x<2; x++){
        for(int y=0; y<2; y++){
            for(int z=0; z<2; z++){
                computed.push_back(libmorton::morton3D_32_encode(x, y, z));
            }
        }
    }
    ASSERT_EQ(gt, computed);
}

TEST(LEAF_TO_MORTON, TEST_ALL_MORTON_INDEXES){
    ASSERT_EQ(LEAF_MASK_TO_MORTON_INDEX_2x2x2(0b10000000), 0);
    ASSERT_EQ(LEAF_MASK_TO_MORTON_INDEX_2x2x2(0b01000000), 4);
    ASSERT_EQ(LEAF_MASK_TO_MORTON_INDEX_2x2x2(0b00100000), 2);
    ASSERT_EQ(LEAF_MASK_TO_MORTON_INDEX_2x2x2(0b00010000), 6);
    ASSERT_EQ(LEAF_MASK_TO_MORTON_INDEX_2x2x2(0b00001000), 1);
    ASSERT_EQ(LEAF_MASK_TO_MORTON_INDEX_2x2x2(0b00000100), 5);
    ASSERT_EQ(LEAF_MASK_TO_MORTON_INDEX_2x2x2(0b00000010), 3);
    ASSERT_EQ(LEAF_MASK_TO_MORTON_INDEX_2x2x2(0b00000001), 7);
}

TEST(SVOLEAF, SetLEAF){
    uint32_t a = 0xFF;
    std::cout << PRINT_BITS_INT32(a) << std::endl;
    a = 0xFF00;
    std::cout << PRINT_BITS_INT32(a) << std::endl;
    a = 0x10000;
    std::cout << PRINT_BITS_INT32(a) << std::endl;

    std::cout << "=====" << std::endl;

    std::cout << "full leaf mask: \t\t\t\t" << PRINT_BITS_INT32(0x000000FF) << std::endl;
    std::cout << "full valid mask: \t\t\t\t" << PRINT_BITS_INT32(0x0000FF00) << std::endl;
    std::cout << "child pointer max value: \t\t" << PRINT_BITS_INT32(0x00007FFF) << std::endl;
    std::cout << "=====" << std::endl;
    std::cout << PRINT_BITS_INT32(12) << " " << IS_CHILD_POINTER_TOO_LARGE(12) << std::endl;
    std::cout << PRINT_BITS_INT32(213) << " " << IS_CHILD_POINTER_TOO_LARGE(213) << std::endl;
    std::cout << PRINT_BITS_INT32(32098) << " " << IS_CHILD_POINTER_TOO_LARGE(32098) << std::endl;
    std::cout << PRINT_BITS_INT32(34980) << " " << IS_CHILD_POINTER_TOO_LARGE(34980) << std::endl;
    std::cout << "=====" << std::endl;

//    std::cout << PRINT_BITS_INT32((0xFF << 24)) << std::endl;
//    std::cout << PRINT_BITS_INT32((0xFF << 16)) << std::endl;

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}