
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include "../../include/stb_image.h"
#include "../utils/Utils.hpp"
#include "../model/ShaderPipeline.hpp"
#include "SkyBox.hpp"

SkyBox::SkyBox(std::shared_ptr<ShaderPipeline> pipeline, const std::vector<std::string> &faces)
{
    m_pipeline = pipeline;

    
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertices), &m_vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glGenTextures(1, &m_textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_textureID);
    int width, height, nrChannels;
    for (unsigned int i = 0; i < faces.size(); i++)
    {
        unsigned char *data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);

            std::cout << "Skybox: Loading cubemap face: " << i << " success." << std::endl;
        }
        else
        {
            std::cout << "Skybox: failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    m_viewLocation = m_pipeline->AddUniform("view");
    m_projectionLocation = m_pipeline->AddUniform("projection");
}

SkyBox::~SkyBox()
{
}

void SkyBox::Render(glm::mat4 projectionMatrix, glm::mat4 viewMatrix)
{
    glDepthMask(GL_FALSE);
    m_pipeline->UseShader();

    glm::mat4 tmp_view = glm::mat4(glm::mat3(viewMatrix));

    glUniformMatrix4fv(m_viewLocation, 1, GL_FALSE, glm::value_ptr(tmp_view));
    glUniformMatrix4fv(m_projectionLocation, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    glBindVertexArray(m_VAO);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_textureID);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glDepthMask(GL_TRUE);
}