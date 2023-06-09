#include <GL/glew.h>
#include <glm/glm.hpp>
#include <memory>
#include "../utils/Utils.hpp"
#include "../utils/SceneSettings.hpp"

#include "../controllers/Scene/Scene.hpp"

#include "Grid.hpp"
#include <iostream>

Grid::Grid(std::shared_ptr<ShaderPipeline> pipeline){
    m_pipeline = pipeline;
    // mVBO = Utils::CreateVertexBuffer(mVertices);
    // mVAO = Utils::CreateObjectBuffer();

    // mViewLocation = mPipeline->AddUniform("view");
    // mProjLocation = mPipeline->AddUniform("proj");
    m_scaleLocation = m_pipeline->AddUniform("scale");
    m_dimensionLocation = m_pipeline->AddUniform("dims");
    m_mvpLocation = m_pipeline->AddUniform("mvp");

    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertices), m_vertices, GL_STATIC_DRAW);

    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

}

void Grid::SetShaderPipeline(std::shared_ptr<ShaderPipeline> pipeline){
    m_pipeline = pipeline;
}


// void Grid::Render(const glm::mat4& projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene){
void Grid::Render(Scene* scene){
    m_pipeline->UseShader();
    
    glm::mat4 ModelMatrix = glm::mat4(1.0);
    glm::mat4 MVP = scene->GetActiveCam()->GetProjectionMatrix() * scene->GetActiveCam()->GetViewMatrix() * ModelMatrix;

    
    glUniform1f(m_scaleLocation, 4.0f);
    glUniform2f(m_dimensionLocation, scene->GetSceneSettings()->GetViewportWidth(), scene->GetSceneSettings()->GetViewportHeight());
    // glUniformMatrix4fv(mProjLocation, 1, GL_FALSE, &projectionMatrix[0][0]);
    // glUniformMatrix4fv(mViewLocation, 1, GL_FALSE, &viewMatrix[0][0]);
    glUniformMatrix4fv(m_mvpLocation, 1, GL_FALSE, &MVP[0][0]);

    glBindVertexArray(m_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 1*6*3);
}