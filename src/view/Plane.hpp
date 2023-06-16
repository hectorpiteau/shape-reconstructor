#pragma once
#include <memory>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Renderable/Renderable.hpp"
#include "../model/ShaderPipeline.hpp"
#include "../model/Texture2D.hpp"
#include "../utils/Utils.hpp"
#include "../controllers/Scene/Scene.hpp"

using namespace glm;
class Scene;

class Plane : Renderable
{
public:
    explicit Plane(Scene* scene);
    Plane(const Plane &) = delete;
    ~Plane() override = default;
    
    void SetTexture2D(Texture2D* texture);

    void SetVertices(const vec3 &top_left, const vec3 &top_right, const vec3 &bot_left, const vec3 &bot_right);

    void SetCustomTex(GLuint tex);
    void SetUseCustomTex(bool use);

    void Render() override;

private:
    /** out dep. */
    Scene *m_scene;

    GLuint m_customTex;
    bool m_useCustomTex;

    mat4 m_model = mat4(1.0);

    ShaderPipeline m_pipeline;
    
    Texture2D* m_texture = nullptr;

    unsigned int m_VBO{};
    unsigned int m_VAO{};

    /** Uniforms */
    GLint m_modelLocation;
    GLint m_viewLocation;
    GLint m_projectionLocation;

    size_t m_size = 6 * 5;

    /** world pos (x,y,z) + tex (u, v)*/
    float m_vertices[6 * 5] = {
        -1.0f, 1.0f, 0.0f,      0.0f, 0.0f,  //top_left
         1.0f, 1.0f, 0.0f,      1.0f, 0.0f,  //top_right
        -1.0f, -1.0f, 0.0f,     0.0f, 1.0f,  //bot_left

         1.0f, 1.0f, 0.0f,      1.0f, 0.0f,  //top_right
         1.0f, -1.0f, 0.0f,     1.0f, 1.0f,  //bot_right
        -1.0f, -1.0f, 0.0f,     0.0f, 1.0f   //bot_left
    };
};