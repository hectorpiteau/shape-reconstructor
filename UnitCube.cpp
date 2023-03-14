#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include "UnitCube.hpp"
#include "Utils.hpp"

UnitCube::UnitCube(std::shared_ptr<ShaderPipeline> pipeline){
    m_pipeline = pipeline;
    glEnable(GL_CULL_FACE);
    // glFrontFace(GL_CW);
    // m_VBO = Utils::CreateVertexBuffer(m_vertices);
    // m_VAO = Utils::CreateObjectBuffer();

    // m_scaleLocation = m_pipeline->AddUniform("scale");
    m_modelLocation = m_pipeline->AddUniform("model");
    m_viewLocation = m_pipeline->AddUniform("view");
    m_projectionLocation = m_pipeline->AddUniform("projection");
    // mDimensionLocation = m_pipeline->AddUniform("dims");
    // mMvpLocation = m_pipeline->AddUniform("mvp");

    m_cameraPos = m_pipeline->AddUniform("viewPos");

    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertices), m_vertices, GL_STATIC_DRAW);

    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (GLvoid*)(sizeof(float) * 3));
    glEnableVertexAttribArray(1);

}

UnitCube::~UnitCube(){
    glDeleteBuffers(1, &m_VBO);
    glDeleteVertexArrays(1, &m_VAO);
}

void UnitCube::SetShaderPipeline(std::shared_ptr<ShaderPipeline> pipeline){
    m_pipeline = pipeline;
}

void UnitCube::Render(glm::mat4& projectionMatrix, glm::mat4& viewMatrix, glm::vec3 cameraPos, float windowWidth, float windowHeight){
    m_pipeline->UseShader();
    
    glm::mat4 model = glm::mat4(1.0);
    model = glm::translate(model, glm::vec3(0.0, 0.5, 0.0));
    

    glUniformMatrix4fv(m_modelLocation, 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(m_viewLocation, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(m_projectionLocation, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    glUniform3f(m_cameraPos, cameraPos.x, cameraPos.y, cameraPos.z);

    glBindVertexArray(m_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6*6*3);

}