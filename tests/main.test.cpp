#include <gtest/gtest.h>
#include <libmorton/morton.h>
#include "../src/maths/Morton.h"
#include <iostream>

TEST(MyClassTest, InitializationTest) {
    std::vector<uint> gt = {0, 4, 32, 36, 2, 6, 34, 38, 16, 20, 48, 52, 18, 22, 50, 54, 1, 5, 33, 37, 3, 7, 35, 39, 17, 21, 49, 53, 19, 23, 51, 55, 8, 12, 40, 44, 10, 14, 42, 46, 24, 28, 56, 60, 26, 30, 58, 62, 9, 13, 41, 45, 11, 15, 43, 47, 25, 29, 57, 61, 27, 31, 59, 63};
    std::vector<uint> computed = {};
    for(int x=0; x<4; x++){
        for(int y=0; y<4; y++){
            for(int z=0; z<4; z++){
                 computed.push_back(libmorton::morton3D_32_encode(x, y, z));
            }
        }
    }
    ASSERT_EQ(gt, computed);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}