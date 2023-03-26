#pragma once
#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <memory>
#include "../model/ShaderPipeline.hpp"
#include "Renderable/Renderable.hpp"

class Lines : Renderable
{
public:
    Lines(const float *data, int dataLength)
        : m_data(data),
          m_dataLength(dataLength),
          m_pipeline("../src/shaders/v_lines.glsl", "../src/shaders/f_lines.glsl")
    {
        m_mvpLocation = m_pipeline.AddUniform("mvp");
        m_colorLocation = m_pipeline.AddUniform("color");

        glGenBuffers(1, &m_VBO);
        glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
        glBufferData(GL_ARRAY_BUFFER, m_dataLength * sizeof(float), m_data, GL_STREAM_DRAW);

        glGenVertexArrays(1, &m_VAO);
        glBindVertexArray(m_VAO);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
        glEnableVertexAttribArray(0);

        m_model = glm::mat4(1.0f);
        m_model = glm::translate(m_model, glm::vec3(0.0, 0.5, 0.0));

        m_ready = true;
    }

    Lines() : m_data(nullptr), m_dataLength(0), m_pipeline("../src/shaders/v_lines.glsl", "../src/shaders/f_lines.glsl")
    {
        m_mvpLocation = m_pipeline.AddUniform("mvp");

        // glGenBuffers(1, &m_VBO);
        // glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
        // glBufferData(GL_ARRAY_BUFFER, 0 * sizeof(float), (void*)0, GL_STREAM_DRAW);

        // glGenVertexArrays(1, &m_VAO);
        // glBindVertexArray(m_VAO);
        // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
        // glEnableVertexAttribArray(0);

        m_model = glm::mat4(1.0f);
        m_model = glm::translate(m_model, glm::vec3(0.0, 0.5, 0.0));

        m_ready = false;
    }

    ~Lines()
    {
        glDeleteVertexArrays(1, &m_VAO);
        glDeleteBuffers(1, &m_VBO);
    }

    /**
     * @brief Update the vertices that defines the lines displayed on screen.
     * TODO: CHECK AND UPDATE TO HANDLE float*.
     *
     * @param vertices : a vector of glm::vec3.
     */
    void UpdateVertices(const std::vector<glm::vec3> &vertices)
    {
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), &vertices[0]);
    }

    /**
     * @brief Set the color of the lines.
     *
     * @param r : Red   value [0,1]
     * @param g : Green value [0,1]
     * @param b : Blue  value [0,1]
     * @param b : Alpha value [0,1]
     */
    void SetColor(double r, double g, double b, double a)
    {
        m_color.x = r;
        m_color.y = g;
        m_color.z = b;
        m_color.w = a;
    }

    void SetVisibleDataLength(int length)
    {
        m_visibleDataLength = length;
    }

    /**
     * @brief Set the color of the lines.
     *
     * @param color : glm::vec4 containing values for (red, green, blue, alpha) all in range [0,1].
     */
    void SetColor(const glm::vec4 &color)
    {
        m_color.x = color.x;
        m_color.y = color.y;
        m_color.z = color.z;
        m_color.w = color.w;
    }

    /**
     * @brief Render lines on screen.
     *
     * @param projectionMatrix : The current camera projection matrix.
     * @param viewMatrix : The current camera view matrix.
     */
    void Render(const glm::mat4 &projectionMatrix, const glm::mat4 &viewMatrix)
    {
        if (!m_ready)
            return;

        m_pipeline.UseShader();
        glm::mat4 MVP = projectionMatrix * viewMatrix * m_model;

        /** Subscribe color vec4. */
        glUniform4fv(m_colorLocation, 1, glm::value_ptr(m_color));
        /** Subscribe MVP matrix. */
        glUniformMatrix4fv(m_mvpLocation, 1, GL_FALSE, &MVP[0][0]);

        glBindVertexArray(m_VAO);

        glDrawArrays(GL_LINES, 0, m_dataLength); // previous length: m_dataLength
    }
    void Render(const glm::mat4 &projectionMatrix, const glm::mat4 &viewMatrix, std::shared_ptr<SceneSettings> scene)
    {
        if (!m_ready)
            return;

        m_pipeline.UseShader();
        glm::mat4 MVP = projectionMatrix * viewMatrix * m_model;

        /** Subscribe color vec4. */
        glUniform4fv(m_colorLocation, 1, glm::value_ptr(m_color));
        /** Subscribe MVP matrix. */
        glUniformMatrix4fv(m_mvpLocation, 1, GL_FALSE, &MVP[0][0]);

        glBindVertexArray(m_VAO);

        glDrawArrays(GL_LINES, 0, m_dataLength); // previous length: m_dataLength
    }

private:
    /** Vertex Buffer Object and Vertex Attribute Object identifiers. */
    unsigned int m_VBO, m_VAO;

    /** Model matrix of the set of lines. Used to compute the MVP matrix. */
    glm::mat4 m_model;

    /** Location for MVP matrix sent to the shader. */
    GLint m_mvpLocation;

    /** Location for color vector sent to the shader. */
    GLint m_colorLocation;

    /** Lines's color. */
    glm::vec4 m_color = glm::vec4(0.7, 0.6, 1.0, 1.0);

    /** The Lines's shader pipeline. */
    ShaderPipeline m_pipeline;

    /** A constant pointer to the data. TODO: clear */
    const float *m_data;
    int m_dataLength;

    int m_visibleDataLength;

    const std::vector<glm::vec3> m_vertices;

    bool m_ready;
};