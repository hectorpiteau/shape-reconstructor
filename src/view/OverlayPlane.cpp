#include <GL/glew.h>
#include <glm/glm.hpp>
#include <memory>
#include "OverlayPlane.hpp"
#include "../model/ShaderPipeline.hpp"
#include "../model/Texture2D.hpp"

OverlayPlane::OverlayPlane(std::shared_ptr<ShaderPipeline> pipeline) : m_pipeline(pipeline)
{
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

    m_texture0 = std::make_shared<Texture2D>(1080, 720, 4, 1);

}

OverlayPlane::~OverlayPlane()
{
}

void OverlayPlane::Render(bool useTex, GLuint tex)
{
    m_pipeline->UseShader();

    glUniform1f(m_scaleLocation, 1.0f);
    if(useTex){
        glBindTexture(GL_TEXTURE_2D, tex);
    }else{
        glBindTexture(GL_TEXTURE_2D, m_texture0->GetID());
    }
    
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 1 * 6 * 3);
}

void OverlayPlane::SetTextureData(const unsigned char *data)
{
}