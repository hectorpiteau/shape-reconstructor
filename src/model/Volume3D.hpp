#ifndef VOLUME_3D
#define VOLUME_3D

#include <memory>
#include <iostream>

#include "../view/Renderable/Renderable.hpp"
#include "../view/Wireframe/Wireframe.hpp"
#include "../view/Lines.hpp"
#include "../view/SceneObject/SceneObject.hpp"

#include "../controllers/Scene/Scene.hpp"

/**
 * @brief This class allows for the render of 3D volumetric data.
 * It is physically contained inside the unit-cube (size 1 in all directions).
 * 
 * The data and processing for ray marching is dispatched to a CUDA kernel.
 * 
 * This class performs the rendering of the result of the CUDA kernel in a plane that overlay the view. 
 */
class Volume3D : public SceneObject, public Wireframe
{
public:
    Volume3D(Scene* scene);

    // void Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene);
    void Render();
    
    void UpdateWireFrame(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene);

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


#endif //VOLUME_3D