#include <memory>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Renderable/Renderable.hpp"
#include "../model/Texture2D.hpp"
#include "../model/ShaderPipeline.hpp"
#include "../utils/Utils.hpp"
#include "../controllers/Scene/Scene.hpp"

#include "Plane.hpp"

using namespace glm;

Plane::Plane(Scene *scene) : m_scene(scene), m_pipeline("../src/shaders/v_plane.glsl", "../src/shaders/f_plane.glsl") {

    m_modelLocation = m_pipeline.AddUniform("model");
    m_viewLocation = m_pipeline.AddUniform("view");
    m_projectionLocation = m_pipeline.AddUniform("projection");

    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertices), m_vertices, GL_STREAM_DRAW);

    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid *) (sizeof(float) * 3));
    glEnableVertexAttribArray(1);
}

void Plane::SetVertices(const vec3 &top_left, const vec3 &top_right, const vec3 &bot_left, const vec3 &bot_right) {
    WRITE_VEC3(m_vertices, 0, top_left);
    WRITE_VEC3(m_vertices, 5, top_right);
    WRITE_VEC3(m_vertices, 10, bot_left);

    WRITE_VEC3(m_vertices, 15, top_right);
    WRITE_VEC3(m_vertices, 20, bot_right);
    WRITE_VEC3(m_vertices, 25, bot_left);



    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, (int) (m_size * sizeof(float)), m_vertices, GL_STREAM_DRAW);
}

void Plane::Render() {
    glDisable(GL_CULL_FACE);
    m_pipeline.UseShader();

    glActiveTexture(GL_TEXTURE0);
    if (m_useCustomTex) {
        glBindTexture(GL_TEXTURE_2D, m_customTex);
    } else {
        glBindTexture(GL_TEXTURE_2D, m_texture->GetID());
    }

    glUniformMatrix4fv(m_modelLocation, 1, GL_FALSE, value_ptr(m_model));
    glUniformMatrix4fv(m_viewLocation, 1, GL_FALSE, value_ptr(m_scene->GetActiveCam()->GetViewMatrix()));
    glUniformMatrix4fv(m_projectionLocation, 1, GL_FALSE, value_ptr(m_scene->GetActiveCam()->GetProjectionMatrix()));

    glBindVertexArray(m_VAO);

    glDrawArrays(GL_TRIANGLES, 0, (int) m_size/5);
    glEnable(GL_CULL_FACE);
}

void Plane::SetTexture2D(Texture2D *texture) {
    m_texture = texture;
}

void Plane::SetCustomTex(GLuint tex) {
    m_customTex = tex;
}

void Plane::SetUseCustomTex(bool use) {
    m_useCustomTex = use;
}
