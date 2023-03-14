#ifndef GRID_H
#define GRID_H
#include <GL/gl.h>
#include <glm/glm.hpp>
#include <memory>
#include "ShaderPipeline.hpp"

class Grid
{
public:
    /**
     * @brief Construct a new Grid object
     * 
     * @param pipeline 
     */
    Grid(std::shared_ptr<ShaderPipeline> pipeline);
    Grid(const Grid&) = delete;

    /**
     * @brief Set the Shader Pipeline object
     * 
     * @param pipeline 
     */
    void SetShaderPipeline(std::shared_ptr<ShaderPipeline> pipeline);

    /**
     * @brief Render the Unit Cube in the scene.
     * 
     * @param projectionMatrix 
     * @param viewMatrix 
     * @return true 
     * @return false 
     */
    void Render(glm::mat4& projectionMatrix, glm::mat4& viewMatrix, float windowWidth, float windowHeight);

private:
    std::shared_ptr<ShaderPipeline> m_pipeline;
    unsigned int m_VBO;
    unsigned int m_VAO;

    /** Uniforms */
    GLint m_scaleLocation;
    GLint m_dimensionLocation;
    GLint m_mvpLocation;
    GLint m_viewLocation;
    GLint m_projLocation;

    /** Uniforms */
    // GLint mProjLocation;
    // GLint mViewLocation;
    // GLint mModelLocation;
    // GLint mMVPLocation;
    // GLint mDimsLocation;
    
    float m_vertices[6*6*6] = {
        // -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, // right
        // 0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f,
        // 0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f,
        // 0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f,
        // -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f,
        // -0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f,

        // -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, // left
        // 0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        // 0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        // 0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        // -0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        // -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f,

        // -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f, // back
        // -0.5f, 0.5f, -0.5f, -1.0f, 0.0f, 0.0f,
        // -0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f,
        // -0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f,
        // -0.5f, -0.5f, 0.5f, -1.0f, 0.0f, 0.0f,
        // -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f,

        // 0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, // front
        // 0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
        // 0.5f, 0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
        // 0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
        // 0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f,
        // 0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 0.0f,

        // -0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, // bot
        // 0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f,
        // 0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f,
        // 0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f,
        // -0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f,
        // -0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f,

        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, // top
        0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
        -0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f
        };
};

#endif // GRID_H