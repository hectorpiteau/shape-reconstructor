//
// Created by hepiteau on 15/06/24.
//
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Rectangle.hpp"

Rectangle::Rectangle(Scene* scene)
: m_scene(scene), m_pipeline("../src/shaders/v_rect.glsl", "../src/shaders/f_rect.glsl")
{
    float vertices[6*3] = {
            0.5f, 0.5f, 0.0f,  // top right
            0.5f, -0.5f, 0.0f,  // bottom right
            -0.5f, 0.5f, 0.0f,  // top left
            // second triangle
            0.5f, -0.5f, 0.0f,  // bottom right
            -0.5f, -0.5f, 0.0f,  // bottom left
            -0.5f, 0.5f, 0.0f   // top left
    };
    m_size = 6*3;
//    m_vertices = (float*)malloc(sizeof(float) * m_size);
    memcpy(m_vertices, vertices, sizeof(float)*6*3);
    Initialize();
}

Rectangle::Rectangle(Scene *scene, const vec3 &min, const vec3 &max)
        : m_scene(scene), m_pipeline("../src/shaders/v_rect.glsl", "../src/shaders/f_rect.glsl")
        {
            float vertices[] = {
                    min.x, min.y, min.z,
                    min.x, max.y, min.z,
                    max.x, max.y, min.z,
                    min.x, min.y, min.z,
                    max.x, max.y, min.z,
                    max.x, min.y, min.z,

                    min.x, min.y, max.z,
                    min.x, max.y, max.z,
                    max.x, max.y, max.z,
                    min.x, min.y, max.z,
                    max.x, max.y, max.z,
                    max.x, min.y, max.z,

                    min.x, min.y, min.z,
                    min.x, min.y, max.z,
                    max.x, min.y, max.z,
                    min.x, min.y, min.z,
                    max.x, min.y, max.z,
                    max.x, min.y, min.z,

                    min.x, max.y, min.z,
                    min.x, max.y, max.z,
                    max.x, max.y, max.z,
                    min.x, max.y, min.z,
                    max.x, max.y, max.z,
                    max.x, max.y, min.z,

                    min.x, min.y, min.z,
                    min.x, max.y, min.z,
                    min.x, max.y, max.z,
                    min.x, min.y, min.z,
                    min.x, max.y, max.z,
                    min.x, min.y, max.z,

                    max.x, min.y, min.z,
                    max.x, max.y, min.z,
                    max.x, max.y, max.z,
                    max.x, min.y, min.z,
                    max.x, max.y, max.z,
                    max.x, min.y, max.z,
            };
    m_size = 6 * 6 * 3;
//    m_vertices = new float[m_size];
    memcpy(m_vertices, vertices, sizeof(float) * m_size);

    float lines_vertices[] = {
            min.x,min.y,min.z,
            min.x,max.y,min.z,

            min.x,min.y,min.z,
            max.x,min.y,min.z,

            min.x,min.y,min.z,
            min.x,min.y,max.z,

            min.x,max.y,min.z,
            max.x,max.y,min.z,

            max.x,max.y,min.z,
            max.x,min.y,min.z,
            //
            min.x,max.y,min.z,
            min.x,max.y,max.z,

            max.x,max.y,min.z,
            max.x,max.y,max.z,

            max.x,min.y,min.z,
            max.x,min.y,max.z,

            min.x,min.y,max.z,
            max.x,min.y,max.z,

            min.x,min.y,max.z,
            min.x,max.y,max.z,

            min.x,max.y,max.z,
            max.x,max.y,max.z,

            max.x,max.y,max.z,
            max.x,min.y,max.z,


    };
    m_lines = std::make_unique<Lines>(scene, lines_vertices, 72, true);

    Initialize();
}

Rectangle::~Rectangle() {
    free(m_vertices);
}


void Rectangle::Render() {
        if(m_lines) m_lines->Render();
        glDisable(GL_CULL_FACE);
        m_pipeline.UseShader();

        glm::mat4 model = glm::mat4(1.0);

        glUniformMatrix4fv(m_modelLocation, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(m_viewLocation, 1, GL_FALSE, glm::value_ptr(m_scene->GetActiveCam()->GetViewMatrix()));
        glUniformMatrix4fv(m_projectionLocation, 1, GL_FALSE, glm::value_ptr(m_scene->GetActiveCam()->GetProjectionMatrix()));
        glUniform4fv(m_colorLocation, 1, (&m_faceColor[0]));

        glBindVertexArray(m_VAO);
        glDrawArrays(GL_TRIANGLES, 0, m_size/3);
        glEnable(GL_CULL_FACE);
}

void Rectangle::Initialize() {
    m_modelLocation = m_pipeline.AddUniform("model");
    m_viewLocation = m_pipeline.AddUniform("view");
    m_projectionLocation = m_pipeline.AddUniform("projection");
    m_colorLocation = m_pipeline.AddUniform("color");

    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertices), m_vertices, GL_STREAM_DRAW);

    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);

}

