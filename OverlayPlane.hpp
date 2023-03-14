#ifndef OVERLAY_PLANE_H
#define OVERLAY_PLANE_H
#include <memory>
#include "ShaderPipeline.hpp"
#include "Texture2D.hpp"

class OverlayPlane
{
public:
    OverlayPlane(std::shared_ptr<ShaderPipeline> pipeline);
    ~OverlayPlane();

    void Render(bool useTex, GLuint tex);

    void SetTextureData(const unsigned char *data);

private:
    int m_width;
    int m_height;
    int m_nbChannels;

    std::shared_ptr<Texture2D> m_texture0;

    float m_scale; /** [0, 1] */

    /** Simple plane geometry */
    float m_vertices[6*5] = {
        -1.0f, 1.0f, 0.0f, 0.0f, 1.0f,      //top-left
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,     //bot-left
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,      //bot-right

        -1.0f, 1.0f, 0.0f, 0.0f, 1.0f,     //top-left
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,     //bot-right
        1.0f, 1.0f, 0.0f, 1.0f, 1.0f        //top-right
    };

    const unsigned char *m_data;

    std::shared_ptr<ShaderPipeline> m_pipeline;
    unsigned int m_VBO;
    unsigned int m_VAO;

    GLint m_scaleLocation;
};

#endif // OVERLAY_PLANE_H