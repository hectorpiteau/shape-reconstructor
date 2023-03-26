#pragma once

#include "Lines.hpp"
#include "Wireframe/Wireframe.hpp"
#include "Renderable/Renderable.hpp"
#include "../utils/Utils.hpp"
#include <memory>
#include <glm/glm.hpp>
#include <iostream>

/**
 * @brief Draw a wireframe grid in the scene.
 * This object is using opengl lines.
 */
class LineGrid : Wireframe, Renderable
{
public:
    LineGrid()
    {
        /** One line per cell in each direction, minus one for each axis (x-axis and z-axis). */
        // m_centerVerticesLength = (int(m_width / m_xCellSize) - 1 + int(m_width / m_zCellSize) - 1 - 2 - 2) * 3;
        m_centerVerticesLength = 16*2*3;

        m_centerVertices = new float[m_centerVerticesLength];

        /** Fill the correct values in the list of vertices. */
        glm::vec3 tmp = glm::vec3(0.0);

        /** Border lines. */
        WRITE_VEC3(m_borderVertices, 0, glm::vec3(-m_width / 2.0f, 0.0f, -m_width / 2.0f));
        WRITE_VEC3(m_borderVertices, 3, glm::vec3(m_width / 2.0f, 0.0f, -m_width / 2.0f));
        WRITE_VEC3(m_borderVertices, 6, glm::vec3(-m_width / 2.0f, 0.0f, m_width / 2.0f));
        WRITE_VEC3(m_borderVertices, 9, glm::vec3(m_width / 2.0f, 0.0f, m_width / 2.0f));
        WRITE_VEC3(m_borderVertices, 12, glm::vec3(-m_width / 2.0f, 0.0f, -m_width / 2.0f));
        WRITE_VEC3(m_borderVertices, 15, glm::vec3(-m_width / 2.0f, 0.0f, m_width / 2.0f));
        WRITE_VEC3(m_borderVertices, 18, glm::vec3(m_width / 2.0f, 0.0f, -m_width / 2.0f));
        WRITE_VEC3(m_borderVertices, 21, glm::vec3(m_width / 2.0f, 0.0f, m_width / 2.0f));

        /** x-axis lines */
        WRITE_VEC3(m_xVertices, 0, glm::vec3(-m_width / 2.0f, 0.0f, 0.0f));
        WRITE_VEC3(m_xVertices, 3, glm::vec3(m_width / 2.0f, 0.0f, 0.0f));

        /** z-axis lines */
        WRITE_VEC3(m_zVertices, 0, glm::vec3(0.0f, 0.0f, -m_width / 2.0f));
        WRITE_VEC3(m_zVertices, 3, glm::vec3(0.0f, 0.0f, m_width / 2.0f));

        /** Center lines. */
        int half_amount_of_lines = (int(m_width / m_xCellSize) - 2) / 2;
        std::cout << "half: " << half_amount_of_lines << std::endl;

        float w2 = ((float)m_width) / 2.0f;

        WRITE_VEC3(m_centerVertices, 0, glm::vec3(-w2 + 1* m_xCellSize, 0.0f, -w2));
        WRITE_VEC3(m_centerVertices, 3, glm::vec3(-w2 + 1* m_xCellSize, 0.0f, w2));

        WRITE_VEC3(m_centerVertices, 6, glm::vec3(-w2 + 2* m_xCellSize, 0.0f, -w2));
        WRITE_VEC3(m_centerVertices, 9, glm::vec3(-w2 + 2* m_xCellSize, 0.0f, w2));

        int cpt = 0;
        for (int i = 1; i <= half_amount_of_lines; ++i){
            WRITE_VEC3(m_centerVertices, cpt, glm::vec3(-w2 + i* m_xCellSize, 0.0f, -w2));
            cpt += 3;
            WRITE_VEC3(m_centerVertices, cpt, glm::vec3(-w2 + i* m_xCellSize, 0.0f, w2));
            cpt += 3;
        }

        for (int i = 1; i <= half_amount_of_lines; ++i){
            WRITE_VEC3(m_centerVertices, cpt, glm::vec3(-w2 + (i+half_amount_of_lines+1) * m_xCellSize, 0.0f, -w2));
            cpt += 3;
            WRITE_VEC3(m_centerVertices, cpt, glm::vec3(-w2 + (i+half_amount_of_lines+1)* m_xCellSize, 0.0f, w2));
            cpt += 3;
        }

        for (int i = 1; i <= half_amount_of_lines; ++i){
            WRITE_VEC3(m_centerVertices, cpt, glm::vec3(-w2, 0.0f, -w2 + i* m_xCellSize));
            cpt += 3;
            WRITE_VEC3(m_centerVertices, cpt, glm::vec3(w2, 0.0f, -w2 + i* m_xCellSize));
            cpt += 3;
        }

        for (int i = 1; i <= half_amount_of_lines; ++i){
            WRITE_VEC3(m_centerVertices, cpt, glm::vec3(-w2, 0.0f, -w2 + (i+half_amount_of_lines+1) * m_xCellSize));
            cpt += 3;
            WRITE_VEC3(m_centerVertices, cpt, glm::vec3(w2, 0.0f, -w2 + (i+half_amount_of_lines+1)* m_xCellSize));
            cpt += 3;
        }

        /** Allocates the lines renderers. */
        m_borderLines = std::make_shared<Lines>(m_borderVertices, m_borderVerticesLength);
        m_xLine = std::make_shared<Lines>(m_xVertices, m_xVerticesLength);
        m_zLine = std::make_shared<Lines>(m_zVertices, m_zVerticesLength);
        m_centerLines = std::make_shared<Lines>(m_centerVertices, m_centerVerticesLength);

        /** Propagate all colors to lines renderers. */
        m_borderLines->SetColor(m_borderLinesColor);
        m_xLine->SetColor(m_xLineColor);
        m_zLine->SetColor(m_zLineColor);
        m_centerLines->SetColor(m_centerlinesColor);
    }

    ~LineGrid()
    {
        delete[] m_centerVertices;
    }

    LineGrid(float width, float xCellSize, float zCellSize)
        : m_width(width), m_xCellSize(xCellSize), m_zCellSize(zCellSize)
    {
        LineGrid();
    };

    void UpdateWireFrame(const glm::mat4 &, const glm::mat4 &, std::shared_ptr<SceneSettings> scene){
        /** Don't change over time. */
    }

    void Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene)
    {
        m_xLine->Render(projection, view);
        m_zLine->Render(projection, view);
        m_centerLines->Render(projection, view);
        m_borderLines->Render(projection, view);
    }

private:
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