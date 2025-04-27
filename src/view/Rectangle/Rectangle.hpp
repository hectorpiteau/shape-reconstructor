//
// Created by hepiteau on 15/06/24.
//

#ifndef DRTMCS_RECTANGLE_HPP
#define DRTMCS_RECTANGLE_HPP
#include <glm/glm.hpp>
#include "../../model/ShaderPipeline.hpp"
#include "../../controllers/Scene/Scene.hpp"
class Scene;

class Rectangle {
private:
    Scene *m_scene;

    glm::vec4 m_faceColor {0.3, 0.1, 0.9, 0.05};
    glm::vec4 m_edgeColor {1.0, 1.0, 1.0, 1.0};

    ShaderPipeline m_pipeline;

    /** Uniforms */
    GLint m_modelLocation;
    GLint m_viewLocation;
    GLint m_projectionLocation;
    GLint m_colorLocation;

    unsigned int m_VBO{};
    unsigned int m_VAO{};

    size_t m_size = 6 * 3;

    std::unique_ptr<Lines> m_lines;

    /** world pos (x,y,z) */
    float m_vertices[110];
//    float m_vertices[6*3] = {
//            0.5f, 0.5f, 0.0f,  // top right
//            0.5f, -0.5f, 0.0f,  // bottom right
//            -0.5f, 0.5f, 0.0f,  // top left
//            // second triangle
//            0.5f, -0.5f, 0.0f,  // bottom right
//            -0.5f, -0.5f, 0.0f,  // bottom left
//            -0.5f, 0.5f, 0.0f   // top left
//    };

    void Initialize();
public:

    Rectangle(Scene* scene);
    ~Rectangle();

    Rectangle(Scene* scene, const vec3& cube_min, const vec3& cube_max);
    Rectangle(const Rectangle&) = delete;

    void Render();

};


#endif //DRTMCS_RECTANGLE_HPP
