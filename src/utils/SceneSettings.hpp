#ifndef SCENE_SETTINGS_H
#define SCENE_SETTINGS_H
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <glm/glm.hpp>
#include <map>
#include "SceneGlobalVariables.hpp"

enum CameraMovementModel
{
    FPS,
    ARCBALL
};


class SceneSettings
{
private:

    /** Viewport information */
    int m_viewportWidth, m_viewportHeight;

    /** Camera movement model to be used by the active camera. */
    enum CameraMovementModel m_cameraModel = CameraMovementModel::ARCBALL;

    /** Scroll offsets. */
    glm::vec2 m_scrollOffsets = glm::vec2(0.0, 0.0);

    /** speed factors */
    float m_scrollSpeed = 1.0f;
    float m_moveSpeed = 1.0f;

    /** Mouse click status. */
    bool m_mouseLeftClick = false, m_mouseRightClick = false;

    bool m_shiftKey = false;
    bool m_ctrlKey = false;
    bool m_altKey = false;

    bool m_volumeRendering = true;

    /** Keys */
    std::map<int, int> m_keyPressed;

public:
    SceneSettings(int viewportWidth, int viewportHeight);

    enum CameraMovementModel GetCameraModel();
    void SetCameraModel(enum CameraMovementModel model);

    void Scroll(double xOffset, double yOffset);

    glm::vec2 GetScrollOffsets();

    bool GetMouseLeftClick();
    void SetMouseLeftClick(bool value);

    void SetMouseRightClick(bool value);
    bool GetMouseRightClick();

    void SetShiftKey(bool value);
    bool GetShiftKey();

    void SetCtrlKey(bool value);
    bool GetCtrlKey();
    
    void SetAltKey(bool value);
    bool GetAltKey();

    int GetViewportWidth();
    int GetViewportHeight();

    float GetViewportRatio();

    void IncreaseScrollSpeed();
    void DecreaseScrollSpeed();

    void SetVariable(SceneGlobalVariables var, bool value);
    bool GetVariable(SceneGlobalVariables var);

    void SetKey(int key, int action);
    bool IsKeyPressed(int key);
};

#endif // SCENE_SETTINGS_H