
#include <iostream>
#include <GLFW/glfw3.h>
#include <map>
#include "SceneGlobalVariables.hpp"

#include "SceneSettings.hpp"

SceneSettings::SceneSettings(int viewportWidth, int viewportHeight) : m_viewportWidth(viewportWidth),
                                                                      m_viewportHeight(viewportHeight),
                                                                      m_keyPressed() {};

enum CameraMovementModel SceneSettings::GetCameraModel() { return m_cameraModel; }

void SceneSettings::SetCameraModel(enum CameraMovementModel model) {
    m_cameraModel = model;
    std::cout << "Set camera mode: " << model << std::endl;
}

void SceneSettings::Scroll(double xOffset, double yOffset) {
    m_scrollOffsets.x += xOffset * m_scrollSpeed;
    m_scrollOffsets.y += yOffset * m_scrollSpeed;
};

glm::vec2 SceneSettings::GetScrollOffsets() { return m_scrollOffsets; }

bool SceneSettings::GetMouseLeftClick() { return m_mouseLeftClick; }

void SceneSettings::SetMouseLeftClick(bool value) { m_mouseLeftClick = value; }

void SceneSettings::SetMouseRightClick(bool value) { m_mouseRightClick = value; }

bool SceneSettings::GetMouseRightClick() { return m_mouseRightClick; }

void SceneSettings::SetShiftKey(bool value) { m_shiftKey = value; }

bool SceneSettings::GetShiftKey() { return m_shiftKey; }

void SceneSettings::SetCtrlKey(bool value) { m_ctrlKey = value; }

bool SceneSettings::GetCtrlKey() { return m_ctrlKey; }

void SceneSettings::SetAltKey(bool value) { m_altKey = value; }

bool SceneSettings::GetAltKey() { return m_altKey; }

int SceneSettings::GetViewportWidth() { return m_viewportWidth; }

int SceneSettings::GetViewportHeight() { return m_viewportHeight; }

float SceneSettings::GetViewportRatio() { return ((float) (m_viewportWidth)) / ((float) (m_viewportHeight)); }

void SceneSettings::IncreaseScrollSpeed() {
    m_scrollSpeed += 0.01f;
    std::cout << "inc scroll" << std::endl;
}

void SceneSettings::DecreaseScrollSpeed() {
    m_scrollSpeed -= 0.01f;
    std::cout << "dec scroll" << std::endl;
}

void SceneSettings::SetVariable(SceneGlobalVariables var, bool value) {
    switch (var) {
        case SceneGlobalVariables::VOLUME_RENDERING:
            m_volumeRendering = !m_volumeRendering;
            break;
    }
}

bool SceneSettings::GetVariable(SceneGlobalVariables var) {
    switch (var) {
        case SceneGlobalVariables::VOLUME_RENDERING:
            return m_volumeRendering;
    }
    return false;
}

void SceneSettings::SetKey(int key, int action) {
    m_keyPressed[key] = action;
}

bool SceneSettings::IsKeyPressed(int key) {
    auto it = m_keyPressed.find(key);
    if (it == m_keyPressed.end()) {
        return false;
    } else {
        return m_keyPressed[key] == GLFW_PRESS;
    }
}