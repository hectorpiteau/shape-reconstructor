//
// Created by hepiteau on 18/08/23.
//

#include "PointCloud.h"

#include <glm/glm.hpp>


PointCloud::PointCloud(Scene *scene, glm::vec3 *points, size_t length) : SceneObject{std::string("POINTCLOUD"),
                                                                                     SceneObjectTypes::POINTCLOUD},
                                                                         m_scene(scene),
                                                                         m_pipeline("../src/shaders/v_points.glsl",
                                                                                    "../src/shaders/f_points.glsl"),
                                                                         m_points(points), m_length(length) {
    m_mvpLocation = m_pipeline.AddUniform("mvp");

    if (points == NULL) exit(EXIT_FAILURE);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, m_length * sizeof(glm::vec3), m_points, GL_STREAM_DRAW);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *) 0);
    glEnableVertexAttribArray(0);

    m_model = glm::mat4(1.0f);
}

PointCloud::~PointCloud() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

void PointCloud::UpdatePoints(glm::vec3 *points, size_t length) {
    m_length = length;
    m_points = points;

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, m_length * sizeof(glm::vec3), m_points, GL_STREAM_DRAW);
}

//void PointCloud::AddPoints(glm::vec3 *points, size_t length) {
//    m_length += length;
//    m_points = points;
//
//    glBindBuffer(GL_ARRAY_BUFFER, vbo);
//    glBufferData(GL_ARRAY_BUFFER, m_length * sizeof(glm::vec3), m_points, GL_STREAM_DRAW);
//}

void PointCloud::Render() {

    glPointSize(8.0f);

    // Use shader program
    m_pipeline.UseShader();

    glm::mat4 MVP = m_scene->GetActiveCam()->GetProjectionMatrix() * m_scene->GetActiveCam()->GetViewMatrix() * m_model;

    /** Subscribe MVP matrix. */
    glUniformMatrix4fv(m_mvpLocation, 1, GL_FALSE, &MVP[0][0]);

    glBindVertexArray(vao);

    // Draw points
    glDrawArrays(GL_POINTS, 0, m_length);

}

