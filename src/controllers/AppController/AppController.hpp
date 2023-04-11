#pragma once
#include <memory>
#include "../Scene/Scene.hpp"
#include "../../utils/SceneSettings.hpp"

#include "../../view/Volume3D.hpp"
#include "../../view/LineGrid.hpp"
#include "../../view/ImUI/ObjectListView.hpp"
#include "../../view/ImUI/InspectorView.hpp"

#include "../../interactors/ObjectListInteractor.hpp"
#include "../../interactors/SceneObjectInteractor.hpp"

#include "../../model/DenseFloat32Volume.hpp"
#include "../../model/SphereSDF.hpp"
#include "../../model/ImageSet.hpp"
#include "../../model/Calibration/OpenCVCalibrator.hpp"



class AppController
{
public:
    AppController(GLFWwindow *window, std::shared_ptr<SceneSettings>& sceneSettings) : m_window(window), m_sceneSettings(sceneSettings)
    {
        /** Create the Scene */
        m_scene = std::make_shared<Scene>(m_sceneSettings, window);

        /** Fill default Scene. */
        m_scene->Add(std::make_shared<Volume3D>());
        m_scene->Add(std::make_shared<LineGrid>());
        m_scene->Add(std::make_shared<ImageSet>());

        /** Create Views and Interactors */
        m_objectListView = std::make_shared<ObjectListView>();
        // m_inspectorView = std::make_shared<InspectorView>();

        m_sceneObjectInteractor = std::make_shared<SceneObjectInteractor>();
        m_objectListInteractor = std::make_shared<ObjectListInteractor>(m_scene, m_objectListView, m_sceneObjectInteractor);

        m_volume = new DenseFloat32Volume(100);
        SphereSDF::PopulateVolume(m_volume);

        m_calibrator = new OpenCVCalibrator();

    }

    ~AppController(){
        delete m_volume;
    }

    std::shared_ptr<Scene>& GetScene(){
        return m_scene;
    }

    void Render(){
        /** Render visible elements in the scene. */
        m_objectListInteractor->Render();

        m_sceneObjectInteractor->Render();
    }

private:
    GLFWwindow *m_window;
    std::shared_ptr<SceneSettings> m_sceneSettings;
    std::shared_ptr<Scene> m_scene;

    /** Interactors */
    std::shared_ptr<ObjectListInteractor> m_objectListInteractor;
    std::shared_ptr<SceneObjectInteractor> m_sceneObjectInteractor;
    
    /** views */
    std::shared_ptr<ObjectListView> m_objectListView;
    std::shared_ptr<InspectorView> m_inspectorView;

    DenseFloat32Volume* m_volume;

    OpenCVCalibrator* m_calibrator;

};