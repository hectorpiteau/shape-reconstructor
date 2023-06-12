#pragma once
#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <memory>
#include "../model/ShaderPipeline.hpp"
#include "SceneObject/SceneObject.hpp"

#include "../controllers/Scene/Scene.hpp"

class Scene;

class Lines : public SceneObject
{
public:
    /**
     * @brief Construct a new Lines SceneObject.
     * 
     * @param data : A constant pointer to a list of floats defining the lines vertex's points
     * in the world space coordinate.
     * @param dataLength : The amount of floats in the data list.
     */
    Lines(Scene* scene, const float *data, size_t dataLength);

    /**
     * @brief Construct a new Lines SceneObject.
     */
    Lines();

    /** Delete copy constructor. */
    Lines(const Lines&) = delete;

    ~Lines() override;
    
    /**
     * @brief Update the vertices that defines the lines displayed on screen.
     * TODO: CHECK AND UPDATE TO HANDLE float*.
     *
     * @param vertices : a vector of glm::vec3.
     */
    void UpdateVertices(const float* vertices);

    /**
     * @brief Set the color of the lines.
     *
     * @param r : Red   value [0,1]
     * @param g : Green value [0,1]
     * @param b : Blue  value [0,1]
     * @param b : Alpha value [0,1]
     */
    void SetColor(double r, double g, double b, double a);

    void SetVisibleDataLength(int length);
    
    /**
     * @brief Set the color of the lines.
     *
     * @param color : glm::vec4 containing values for (red, green, blue, alpha) all in range [0,1].
     */
    void SetColor(const glm::vec4 &color);

    /**
     * @brief Render lines on screen.
     *
     */
    void Render() override;

private:
    Scene* m_scene;
    /** Vertex Buffer Object and Vertex Attribute Object identifiers. */
    unsigned int m_VBO, m_VAO;

    /** Model matrix of the set of lines. Used to compute the MVP matrix. */
    glm::mat4 m_model;

    /** Location for MVP matrix sent to the shader. */
    GLint m_mvpLocation;

    /** Location for color vector sent to the shader. */
    GLint m_colorLocation;

    /** Lines's color. */
    glm::vec4 m_color = glm::vec4(0.7, 0.6, 1.0, 1.0);

    /** The Lines's shader pipeline. */
    ShaderPipeline m_pipeline;

    /** A constant pointer to the data. TODO: clear */
    const float *m_data;
    size_t m_dataLength;

    int m_visibleDataLength;

    const std::vector<glm::vec3> m_vertices;

    bool m_ready;
};