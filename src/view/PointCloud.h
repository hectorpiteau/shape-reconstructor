//
// Created by hepiteau on 18/08/23.
//

#ifndef DRTMCS_POINTCLOUD_H
#define DRTMCS_POINTCLOUD_H
#include <glm/glm.hpp>
#include "SceneObject/SceneObject.hpp"
#include "../controllers/Scene/Scene.hpp"


class Scene;

class PointCloud : public SceneObject {
private:
    Scene* m_scene;

    /** Model matrix of the set of lines. Used to compute the MVP matrix. */
    glm::mat4 m_model{};

    GLuint vbo;
    GLuint vao;

    /** Location for MVP matrix sent to the shader. */
    GLint m_mvpLocation;

    /** The PointCloud's shader pipeline. */
    ShaderPipeline m_pipeline;

    glm::vec3* m_points;
    size_t m_length;
public:

    PointCloud(Scene* scene, glm::vec3* points, size_t length);
    ~PointCloud();

    void UpdatePoints(glm::vec3* points, size_t length);
    void Render() override;
};


#endif //DRTMCS_POINTCLOUD_H
