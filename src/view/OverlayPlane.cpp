#include <GL/glew.h>
#include <glm/glm.hpp>
#include <memory>
#include "OverlayPlane.hpp"
#include "../model/ShaderPipeline.hpp"
#include "../model/Texture2D.hpp"
#include "../utils/SceneSettings.hpp"
#include "../controllers/Scene/Scene.hpp"

OverlayPlane::OverlayPlane(std::shared_ptr<SceneSettings> sceneSettings){
    m_pipeline = std::make_shared<ShaderPipeline>("../src/shaders/v_overlay_plane.glsl", "../src/shaders/f_overlay_plane.glsl");
    Initialize(sceneSettings->GetViewportWidth(), sceneSettings->GetViewportHeight());
}

OverlayPlane::OverlayPlane(std::shared_ptr<ShaderPipeline> pipeline, std::shared_ptr<SceneSettings> sceneSettings) : m_pipeline(pipeline)
{
    Initialize(sceneSettings->GetViewportWidth(), sceneSettings->GetViewportHeight());
}

void OverlayPlane::Initialize(int width, int height){
    m_scaleLocation = m_pipeline->AddUniform("scale");

    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertices), m_vertices, GL_STATIC_DRAW);

    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)(sizeof(float) * 3));
    glEnableVertexAttribArray(1);

    m_texture0 = std::make_shared<Texture2D>(width, height, 4, 1);
}

OverlayPlane::~OverlayPlane() = default;

void OverlayPlane::Render(bool useTex, GLuint tex)
{
    m_pipeline->UseShader();
    mat4 m_model = mat4(1.0);

    glUniform1f(m_scaleLocation, 1.0f);

    if(useTex){
        glBindTexture(GL_TEXTURE_2D, tex);
    }else{
        glBindTexture(GL_TEXTURE_2D, m_texture0->GetID());
    }
    
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 1 * 6);
}

void OverlayPlane::SetTextureData(const unsigned char *data)
{
}