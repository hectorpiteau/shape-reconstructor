#pragma once

struct OctreeCellData {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

struct OctreeCell {
    /** The depth of this cell in the octree. */
    int depth;

    /** Up to 8 childrens, if nullptr no children. */
    struct OctreeCell* childrens[8] = {nullptr};

    /** Data */
    struct OctreeCellData* data;
    
    /** padding */
    //TODO
};

class Octree  {
public:
    Octree(int maxDepth) : m_maxDepth(maxDepth){
        
    }

private:
    /** Depth 0 = root (full cube)
     * Depth 1: full cube divided by 8.
     * Depth x: ...
    */
    int m_maxDepth = 10;

    struct OctreeCell* m_pool;

    /** Contains the final cells, must be */
    struct OctreeCellData* m_data;

};