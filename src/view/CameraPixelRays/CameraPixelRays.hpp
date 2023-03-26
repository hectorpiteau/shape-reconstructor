#pragma once
#include <memory>
#include "../Lines.hpp"
#include "../Wireframe/Wireframe.hpp"
#include "../Renderable/Renderable.hpp"

class CameraPixelRays : Wireframe, Renderable
{
public:
    /**
     * @brief Construct a new Camera Pixel Rays object. It can be used
     * to visualize rays comming from a camera toward world.
     * Can be used to display a sub-section of the rays, a small window.
     *
     * @param xStride : The jump necessary to go from one ray to another in the x-axis.
     * @param yStride : The jump necessary to go from one ray to another in the x-axis.
     * @param origin : The origin coordinates in pixels from where the rays will be drawn.
     * @param size : The width/height in pixels of the area to draw rays.
     */
    CameraPixelRays(int xStride, int yStride, glm::vec2 origin, glm::vec2 size)
    {
        /** Allocate memory for the patch without considering stride in order to be able to modify it over time. */
        m_verticesFloatCount = size.x * size.y * 3; 
        m_visibleVerticesFloatCount = m_verticesFloatCount;

        m_vertices = new float[m_verticesFloatCount] {0.0f};
        m_lines = std::make_shared<Lines>(m_vertices, m_verticesFloatCount);

    }

    ~CameraPixelRays(){
        delete [] m_vertices;
    }

    void SetXStride(int stride) { m_xStride = stride; }
    void SetYStride(int stride) { m_yStride = stride; }

    int GetXStride() { return m_xStride; }
    int GetYStride() { return m_yStride; }

    void UpdateWireFrame(const glm::mat4 &intrinsics, const glm::mat4 &extrinsics, std::shared_ptr<SceneSettings> scene)
    {
        /** Update the visible vertices. */

        // int cpt = 0;
        // for (int j = 0; j < height; ++j)
        // {
        //     for (int i = 0; i < width; ++i)
        //     {
        //         WRITE_VEC3(vertices2, cpt, pos);
        //         cpt += 3;
        //         glm::vec3 tmp = glm::vec3((-1.0 + i * 2 * wres + wres) * sceneSettings->GetViewportRatio(), -1.0 + j * 2 * hres, -1.0);
        //         glm::vec4 res1 = Projection::CameraToWorld(glm::vec4(tmp, 1.0f), ext, pos);
        //         WRITE_VEC3(vertices2, cpt, res1);
        //         cpt += 3;
        //     }
        // }
    }
    
    /**
     * @brief Render the set of rays on the screen.
     * 
     * @param projection : The camera's projection matrix (intrinsics);
     * @param view : The camera's view matrix (extrinsics);
     * @param scene : The scene's informations.
     */
    void Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene)
    {
        m_lines->SetVisibleDataLength(m_verticesFloatCount);
        m_lines->Render(projection, view);
    }

private:
    /** Stride detones the jump necessary to go from one ray to another.
     * A stride of one will draw every rays. A stride of 2 will draw one ray on two. */
    int m_xStride = 1;

    /** Same stride definition but in the y-axis. */
    int m_yStride = 1;

    float *m_vertices;
    
    int m_verticesFloatCount;
    int m_visibleVerticesFloatCount;

    bool m_isInitialized = false;

    std::shared_ptr<Lines> m_lines;
};