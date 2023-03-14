#pragma once
#include <stdlib.h>
#include <glm/glm.hpp>

struct VoidCell
{
    unsigned int _[6];
};

struct Cell
{
    int x;
};

struct Indirection
{
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;
};

struct Cell VOID_CELL = {
    // ._ = {0, 0, 0, 0, 0, 0}
    .x = -1};

/**
 * @brief Octree, storage Z-shape,
 *
 */
class OctreeD2
{

    OctreeD2()
    {

        m_data = (struct Cell *)malloc(m_dataCellElementSize * m_dataCellSize);
        m_poolStage0 = (struct Indirection *)malloc(m_dataCellElementSize * m_dataCellSize);
        m_poolStage1 = (struct Indirection *)malloc(m_dataCellElementSize * m_dataCellSize);

        /** Init 8 first cells with void cells. */
        for (int i = 0; i < 8; i++)
        {
            m_data[i] = VOID_CELL;
        }
        m_reservedCells = 8;
    }

private:
    int m_dataCellElementSize = sizeof(int);
    int m_dataCellSize = 1;

    int m_indirectionStages = 2;

    int m_indPoolCellSize = 8; /**  **/

    struct Indirection *m_poolStage0;
    // struct Indirection* m_poolStage1;
    struct Cell *m_data;

    int m_reservedCells = 0;

    int m_maxDepth = 12;
    /** Full cube: depth=0
     *  1/8 : depth=1
     *  1/16: depth=2
     */

    bool IsLeaf(int depth, int index)
    {
        if (m_data != nullptr)
        {
            if (m_data + index == 0 && m_data + index + 1 == 0)
        }
    }

    void *GetLeafDataPt()
    {
    }

    void Divide(int depth, int index)
    {
    }

    void AddCell()
    {
    }

//     float4 tree_lookup(uniform sampler3D IndirPool, // Indirection Pool
//                        uniform float3 invS,         // 1 / S
//                        uniform float N, float3 M)   // Lookup coordinates
//     {
//         float4 I = float4(0.0, 0.0, 0.0, 0.0);
//         float3 MND = M;
//         for (float i = 0; i < HRDWTREE_MAX_DEPTH; i++)
//         {
//             // fixed # of iterations
//             float3 P;                                      // compute lookup coords. within current node
//             P = (MND + floor(0.5 + I.xyz * 255.0)) * invS; // access indirection pool
//             if (I.w < 0.9)                                 // already in a leaf?
//                 I = (float4)tex3D(IndirPool, P);           // no, continue to next depth
// #ifdef DYN_BRANCHING                                       // early exit if hardware supports dynamic branching
//             if (I.w > 0.9)                                 // a leaf has been reached
//                 break;
// #endif
//             if (I.w < 0.1) // empty cell
//                 discard;   // compute pos within next depth grid
//             MND = MND * N;
//         }
//         return (I);
//     }

    /**
     * @brief
     *
     */
    bool Traverse(const glm::vec3 position, int depth)
    {

        if (depth == m_maxDepth)
        {
            return false;
        }
        else
        {
            /** Search in indirection table. */
            glm::vec3 tmp = glm::floor(position + glm::vec3(0.5, 0.5, 0.5));
            int index = tmp.x + tmp.y * 2 + tmp.z * 4; 
        }

        return true;
    }
};
