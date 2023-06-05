#ifndef SCENE_SETTINGS_H
#define SCENE_SETTINGS_H
#include <iostream>
#include "SceneGlobalVariables.hpp"
enum CameraMovementModel
{
    FPS,
    ARCBALL
};


class SceneSettings
{
private:

    /** Viewport informations */
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

public:
    SceneSettings(int viewportWidth, int viewportHeight) : m_viewportWidth(viewportWidth), m_viewportHeight(viewportHeight) {};

    enum CameraMovementModel GetCameraModel() { return m_cameraModel; }
    void SetCameraModel(enum CameraMovementModel model)
    {
        m_cameraModel = model;
        std::cout << "Set camera mode: " << model << std::endl;
    }

    void Scroll(double xOffset, double yOffset)
    {
        m_scrollOffsets.x += xOffset * m_scrollSpeed;
        m_scrollOffsets.y += yOffset * m_scrollSpeed;
    };

    glm::vec2 GetScrollOffsets() { return m_scrollOffsets; }

    bool GetMouseLeftClick() { return m_mouseLeftClick; }
    void SetMouseLeftClick(bool value) { m_mouseLeftClick = value; }

    void SetMouseRightClick(bool value) { m_mouseRightClick = value; }
    bool GetMouseRightClick() { return m_mouseRightClick; }

    void SetShiftKey(bool value){ m_shiftKey = value;}
    bool GetShiftKey(){ return m_shiftKey;}

    void SetCtrlKey(bool value){ m_ctrlKey = value;}
    bool GetCtrlKey(){ return m_ctrlKey;}
    
    void SetAltKey(bool value){ m_altKey = value;}
    bool GetAltKey(){ return m_altKey;}

    int GetViewportWidth(){return m_viewportWidth;}
    int GetViewportHeight(){return m_viewportHeight;}

    float GetViewportRatio(){ return ((float)(m_viewportWidth))/((float)(m_viewportHeight));}

    void IncreaseScrollSpeed(){ m_scrollSpeed += 0.01f; std::cout << "inc scroll" << std::endl;}
    void DecreaseScrollSpeed(){ m_scrollSpeed -= 0.01f;std::cout << "dec scroll" << std::endl;}

    void SetVariable(SceneGlobalVariables var, bool value){
        switch(var){
            case SceneGlobalVariables::VOLUME_RENDERING:
                m_volumeRendering = !m_volumeRendering;
                break;
        }
    }

    bool GetVariable(SceneGlobalVariables var){
        switch(var){
            case SceneGlobalVariables::VOLUME_RENDERING:
                return m_volumeRendering;
                break;
            
        }
        return false;

    }
};

#endif // SCENE_SETTINGS_H