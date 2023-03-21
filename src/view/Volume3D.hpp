#pragma once
#include "Renderable/Renderable.hpp"
#include "Wireframe/Wireframe.hpp"
#include "Lines.hpp"
#include <memory>

/**
 * @brief This class allows for the render of 3D volumetric data.
 * It is physically contained inside the unit-cube (size 1 in all directions).
 * 
 * The data and processing for ray marching is dispatched to a CUDA kernel.
 * 
 * This class performs the rendering of the result of the CUDA kernel in a plane that overlay the view. 
 */
class Volume3D : Renderable, Wireframe
{
public:
    Volume3D(){
        m_lines = std::make_shared<Lines>(m_wireframeVertices, 12*2*3);
    }

    void Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene)
    {
        /** nothing special here for now */
    }

    void RenderWireframe(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene){
        m_lines->Render(projection, view);
    }

private: 
    std::shared_ptr<Lines> m_lines;
    float m_wireframeVertices[12*2*3] = {
        -0.5f, -0.5f, -0.5f, // first bottom line x-dir
         0.5f, -0.5f, -0.5f, 
         
        -0.5f, -0.5f,  0.5f, // second bottom line x-dir
         0.5f, -0.5f,  0.5f, 
         
        -0.5f,  0.5f, -0.5f, // first top line x-dir
         0.5f,  0.5f, -0.5f, 
         
        -0.5f,  0.5f,  0.5f, // second top line x-dir
         0.5f,  0.5f,  0.5f, 

        -0.5f,  0.5f, -0.5f, // first back line y-dir  
        -0.5f, -0.5f, -0.5f, 

        -0.5f,  0.5f,  0.5f, // second back line y-dir  
        -0.5f, -0.5f,  0.5f, 

         0.5f,  0.5f, -0.5f, // first front line y-dir  
         0.5f, -0.5f, -0.5f, 
        
         0.5f,  0.5f,  0.5f, // second front line y-dir  
         0.5f, -0.5f,  0.5f,

        -0.5f, -0.5f, -0.5f, // first bottom line z-dir  
        -0.5f, -0.5f,  0.5f,
    
         0.5f, -0.5f, -0.5f, // second bottom line z-dir  
         0.5f, -0.5f,  0.5f,

        -0.5f,  0.5f, -0.5f, // first top line z-dir  
        -0.5f,  0.5f,  0.5f,

         0.5f,  0.5f, -0.5f, // second top line z-dir  
         0.5f,  0.5f,  0.5f,
    };
};