#ifndef OVERLAY_PLANE_H
#define OVERLAY_PLANE_H
#include <memory>
#include "../model/Texture2D.hpp"
#include "../model/ShaderPipeline.hpp"
#include "../utils/SceneSettings.hpp"
#include "../controllers/Scene/Scene.hpp"

class OverlayPlane
{
public:
    explicit OverlayPlane(std::shared_ptr<SceneSettings> sceneSettings);
    OverlayPlane(std::shared_ptr<ShaderPipeline> pipeline,std::shared_ptr<SceneSettings> sceneSettings);
    ~OverlayPlane();

    void Render(bool useTex, GLuint tex, Scene* m_scene);

    void SetTextureData(const unsigned char *data);

private:
    void Initialize(int width, int height);
    
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
    /** Uniforms */
    GLint m_modelLocation;
    GLint m_viewLocation;
    GLint m_projectionLocation;
};

#endif // OVERLAY_PLANE_H