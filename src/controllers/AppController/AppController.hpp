#pragma once
#include <memory>
#include "../Scene/Scene.hpp"
#include "../../utils/SceneSettings.hpp"

#include "../../view/Volume3D.hpp"
#include "../../view/LineGrid.hpp"

#include "../../view/ImUI/ObjectListView.hpp"

#include "../../interactors/ObjectListInteractor.hpp"

class AppController
{
public:
    AppController(GLFWwindow *window, std::shared_ptr<SceneSettings> sceneSettings) : m_window(window), m_sceneSettings(sceneSettings)
    {
        /** Create the Scene */
        m_scene = std::make_shared<Scene>(m_sceneSettings, window);

        /** Fill default Scene. */
        m_scene->Add(std::make_shared<Volume3D>());
        m_scene->Add(std::make_shared<LineGrid>());

        /** Create Views and Interactors */
        m_objectListView = std::make_shared<ObjectListView>();
        m_objectListInteractor = std::make_shared<ObjectListInteractor>(m_scene, m_objectListView);
    }

    std::shared_ptr<Scene>& GetScene(){
        return m_scene;
    }

    void Render(){
        /** Render visible elements in the scene. */
        m_scene->RenderAll();

        /** Render the UI. */
        m_objectListView->Render();
    }

private:
    GLFWwindow *m_window;
    std::shared_ptr<SceneSettings> m_sceneSettings;
    std::shared_ptr<Scene> m_scene;

    /** views */
    std::shared_ptr<ObjectListInteractor> m_objectListInteractor;
    std::shared_ptr<ObjectListView> m_objectListView;
};