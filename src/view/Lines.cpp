#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <memory>

#include "Lines.hpp"

#include "../model/ShaderPipeline.hpp"
#include "Renderable/Renderable.hpp"

Lines::Lines(const float *data, int dataLength)
    : SceneObject {std::string("LINES")},
      m_data(data),
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

Lines::Lines() : SceneObject {std::string("LINES")}, m_data(nullptr), m_dataLength(0), m_pipeline("../src/shaders/v_lines.glsl", "../src/shaders/f_lines.glsl")
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

Lines::~Lines()
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
void Lines::UpdateVertices(const std::vector<glm::vec3> &vertices)
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
void Lines::SetColor(double r, double g, double b, double a)
{
    m_color.x = r;
    m_color.y = g;
    m_color.z = b;
    m_color.w = a;
}

void Lines::SetVisibleDataLength(int length)
{
    m_visibleDataLength = length;
}

/**
 * @brief Set the color of the lines.
 *
 * @param color : glm::vec4 containing values for (red, green, blue, alpha) all in range [0,1].
 */
void Lines::SetColor(const glm::vec4 &color)
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
void Lines::Render(const glm::mat4 &projectionMatrix, const glm::mat4 &viewMatrix)
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

void Lines::Render(const glm::mat4 &projectionMatrix, const glm::mat4 &viewMatrix, std::shared_ptr<SceneSettings> scene)
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
