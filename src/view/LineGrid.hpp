#pragma once

#include "Lines.hpp"
#include "Renderable/Renderable.hpp"
#include "../utils/Utils.hpp"
#include "SceneObject/SceneObject.hpp"
#include "../controllers/Scene/Scene.hpp"
#include <memory>
#include <glm/glm.hpp>
#include <iostream>

class Scene;

/**
 * @brief Draw a wireframe grid in the scene.
 * This object is using opengl lines.
 */
class LineGrid : public SceneObject
{
public:
    LineGrid(Scene* scene, float width, float xCellSize, float zCellSize);
    LineGrid(Scene* scene);
    ~LineGrid();

    void Render();
    void Initialize();

private:
    Scene* m_scene;
    /** The size of the grid. */
    float m_width = 10.0f;

    /** The size of one square on the floor in the world coordinate unit system. */
    float m_xCellSize = 1.0f, m_zCellSize = 1.0f;

    /** The set of (vertice, vertice) that defines the grid lines. */
    float m_borderVertices[8 * 3];
    float m_xVertices[2 * 3];
    float m_zVertices[2 * 3];
    float *m_centerVertices;

    int m_borderVerticesLength = 8 * 3;
    int m_xVerticesLength = 2 * 3;
    int m_zVerticesLength = 2 * 3;
    int m_centerVerticesLength;

    /** Lines rendering object. */
    std::shared_ptr<Lines> m_borderLines;
    std::shared_ptr<Lines> m_xLine;
    std::shared_ptr<Lines> m_zLine;
    std::shared_ptr<Lines> m_centerLines;

    /** Lines's color. */
    glm::vec4 m_centerlinesColor = glm::vec4(1.0, 1.0, 1.0, 0.3);
    glm::vec4 m_borderLinesColor = glm::vec4(1.0, 1.0, 1.0, 0.9);
    glm::vec4 m_xLineColor = glm::vec4(1.0, 0.0, 0.0, 1.0);
    glm::vec4 m_zLineColor = glm::vec4(0.0, 0.0, 1.0, 1.0);
};