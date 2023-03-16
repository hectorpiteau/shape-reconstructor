#pragma once
#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <memory>
#include "../model/ShaderPipeline.hpp"

class Lines
{
public:
    Lines(const float* data, int dataLength)
        : m_data(data),
          m_dataLength(dataLength),
          m_pipeline("../src/shaders/v_lines.glsl", "../src/shaders/f_lines.glsl")
    {
        m_mvpLocation = m_pipeline.AddUniform("mvp");

        glGenBuffers(1, &m_VBO);
        glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
        glBufferData(GL_ARRAY_BUFFER, m_dataLength * sizeof(float), m_data, GL_STREAM_DRAW);
        
        glGenVertexArrays(1, &m_VAO);
        glBindVertexArray(m_VAO);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
        glEnableVertexAttribArray(0);

        m_model = glm::mat4(1.0f);
        m_model = glm::translate(m_model, glm::vec3(0.0, 0.5, 0.0));
    }

    Lines():  m_pipeline("../src/shaders/v_lines.glsl", "../src/shaders/f_lines.glsl"){
        m_mvpLocation = m_pipeline.AddUniform("mvp");
    }

    ~Lines() {
        glDeleteVertexArrays(1, &m_VAO);
        glDeleteBuffers(1, &m_VBO);
    }

    void UpdateVertices(const std::vector<glm::vec3> vertices){
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), &vertices[0]);
    }

    void Render(glm::mat4 &projectionMatrix, glm::mat4 &viewMatrix)
    {
        glLineWidth(10.0f);
        m_pipeline.UseShader();

        glm::mat4 MVP = projectionMatrix * viewMatrix * m_model;
        
        glUniformMatrix4fv(m_mvpLocation, 1, GL_FALSE, &MVP[0][0]);

        glBindVertexArray(m_VAO);
        glDrawArrays(GL_LINES, 0, m_dataLength);
    }

private:
    unsigned int m_VBO, m_VAO;
    glm::mat4 m_model;
    GLint m_modelLocation;
    GLint m_viewLocation;
    GLint m_projectionLocation;
    GLint m_mvpLocation;
    
    // float data[12] = 
    // {
    //     0.0, 0.0, 0.0,
    //     1.0, 1.0, 0.0,
    //     1.0, 1.0, 0.0,
    //     2.0, 1.0, 0.0
    // };

    const float* m_data;
    int m_dataLength;
    const std::vector<glm::vec3> m_vertices;
    ShaderPipeline m_pipeline;
};