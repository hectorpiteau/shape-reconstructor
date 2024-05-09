//
// Created by hepiteau on 09/05/24.
//

#ifndef DRTMCS_MORTON_H
#define DRTMCS_MORTON_H

#include <cstdint> // for uint32_t

namespace Morton {
    // Helper function to interleave bits of a single coordinate with zeros in between them
    uint64_t part1by2(uint32_t x) {
        x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
        x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x ^ (x <<  2)) & 0x09249249; // x = ---- -9-- 8--- --7- -6-- --5- -4-- --3- -2-- --1- -0--
        return x;
    }

    // Function to compute Morton code from 3D coordinates (x, y, z)
    uint64_t mortonCode3D(uint32_t x, uint32_t y, uint32_t z) {
        uint64_t xx = part1by2(x);
        uint64_t yy = part1by2(y);
        uint64_t zz = part1by2(z);
        return (xx * 4) + (yy * 2) + zz;
    }
};


#endif //DRTMCS_MORTON_H
