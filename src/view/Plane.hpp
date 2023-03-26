#pragma once
#include "Renderable/Renderable.hpp"
#include "ShaderPipeline.hpp"
#include "../utils/Utils.hpp"
#include <memory>

class Plane : Renderable {
public:

Plane(std::shared_ptr<ShaderPipeline> pipeline){
    m_modelLocation = m_pipeline->AddUniform("model");
    m_viewLocation = m_pipeline->AddUniform("view");
    m_projectionLocation = m_pipeline->AddUniform("projection");
    m_colorLocation = m_pipeline->AddUniform("color");


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

void SetColor(glm::vec4 color){
    m_color = color;
}

void SetLocation(glm::vec3 pos, glm::vec3 right, glm::vec3 forward){
    m_pos = pos;
    m_right = right;
    m_forward = forward;
    m_up = glm::cross(right, forward);
}

void SetPos(glm::vec3 pos){
    m_pos = pos;
}

void SetForward(glm::vec3 forward){
    m_forward = forward;
}

void SetUp(glm::vec3 up){
    m_up = up;
}

void SetRight(glm::vec3 right){
    m_right = right;
}


void SetVertices(glm::vec3 top_left, glm::vec3 top_right, glm::vec3 bot_left, glm::vec3 bot_right){
    WRITE_VEC3(m_vertices, 0, top_left);
    WRITE_VEC3(m_vertices, 3, top_right);
    WRITE_VEC3(m_vertices, 6, bot_left);

    WRITE_VEC3(m_vertices, 9, top_right);
    WRITE_VEC3(m_vertices, 12, bot_right);
    WRITE_VEC3(m_vertices, 15, bot_left);
}

void Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene){
    m_pipeline->UseShader();

    glUniformMatrix4fv(m_modelLocation, 1, GL_FALSE, glm::value_ptr(m_model));
    glUniformMatrix4fv(m_viewLocation, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(m_projectionLocation, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    glBindVertexArray(m_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6*6*3);
}

private:
    glm::vec3 m_pos;
    glm::vec3 m_right;
    glm::vec3 m_up;
    glm::vec3 m_forward;

    glm::vec4 m_color;

    glm::mat4 m_model = glm::mat4(1.0);

    std::shared_ptr<ShaderPipeline> m_pipeline
    
    unsigned int m_VBO;
    unsigned int m_VAO;

    /** Uniforms */
    GLint m_modelLocation;
    GLint m_viewLocation;
    GLint m_projectionLocation;
    GLint m_colorLocation;
    
    /** world pos (x,y,z) + (u, v)*/
    float* m_vertices[3*6 + 2*6] = {0.0f};
};