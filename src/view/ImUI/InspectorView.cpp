#include <memory>

#include "InspectorView.hpp"

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"

#include "../../../include/icons/IconsFontAwesome6.h"
#include "../SceneObject/SceneObject.hpp"

#include "../../interactors/SceneObjectInteractor.hpp"

#include "Inspectors/ImageSetEditor.hpp"
#include "Inspectors/CameraInspector.hpp"
#include "Inspectors/NeRFInspector.hpp"
#include "Inspectors/VolumeEditor.hpp"
#include "Inspectors/VolumeRendererEditor.hpp"
#include "Inspectors/PlaneCutEditor.hpp"



InspectorView::InspectorView(SceneObjectInteractor* interactor) : m_interactor(interactor) {
    m_imageSetInspectorView = new ImageSetEditor(m_interactor->imageSetInteractor);
    m_cameraInspectorView = new CameraInspector(m_interactor->cameraInteractor);
    m_cameraSetInspectorView = new CameraSetInspector(m_interactor->cameraSetInteractor);
    m_nerfDatasetInspectorView = new NeRFInspector(m_interactor->nerfInteractor);
    m_volumeEditorView = new VolumeEditor(m_interactor->volume3DInteractor);
    m_volumeRendererEditorView = new VolumeRendererEditor(m_interactor->volumeRendererInteractor);
    m_simpleRayCasterEditorView = new SimpleRayCasterEditor(m_interactor->simpleRayCasterInteractor);
    m_planeCutEditorView = new PlaneCutEditor(m_interactor->planeCutInteractor);
    m_adamEditorView = new AdamEditor(m_interactor->adamInteractor);
}

InspectorView::~InspectorView() {
    delete m_imageSetInspectorView;
    delete m_cameraInspectorView;
    delete m_nerfDatasetInspectorView;
}

void InspectorView::SetSelected(enum SceneObjectTypes type){
    m_selectedType = type;
    m_selectedId = m_interactor->GetSceneObjectId();
    m_selectedName = m_interactor->GetSceneObjectName();

    switch(m_interactor->GetSelectedSceneObjectType()){
        case SceneObjectTypes::IMAGESET:
            // m_imageSetInspectorView->SetImageSet(m_interactor->imageSetInteractor->GetImageSet());
            break;
        case SceneObjectTypes::CAMERA:
            // m_cameraInspectorView->SetCamera(m_interactor->cameraInteractor->GetCamera());
            break;
        case SceneObjectTypes::NERFDATASET:
            // m_nerfDatasetInspectorView->SetNeRFDataset();
            break;
        case SceneObjectTypes::NERFDATASET:
            // m_nerfDatasetInspectorView->SetNeRFDataset();
            break;
        case SceneObjectTypes::NONE:
        default:
            ImGui::TextWrapped("Select an object in the object's list to edit its properties here.");
            break;
    }
}

void InspectorView::Render(){
    if(m_interactor == nullptr){
        ImGui::Text("Interactor is null.");
        return;
    }

    ImGui::Begin("Inspector");

    switch(m_interactor->GetSelectedSceneObjectType()){
        case SceneObjectTypes::IMAGESET:
            m_imageSetInspectorView->Render();
            break;
        case SceneObjectTypes::CAMERA:
            m_cameraInspectorView->Render();
            break;
        case SceneObjectTypes::CAMERASET:
            m_cameraSetInspectorView->Render();
            break;
        case SceneObjectTypes::NERFDATASET:
            m_nerfDatasetInspectorView->Render();
            break;
        case SceneObjectTypes::VOLUME3D:
            m_volumeEditorView->Render();
            break;
        case SceneObjectTypes::VOLUMERENDERER:
            m_volumeRendererEditorView->Render();
            break;
        case SceneObjectTypes::RAYCASTER:
            m_simpleRayCasterEditorView->Render();
            break;
        case SceneObjectTypes::PLANECUT:
            m_planeCutEditorView->Render();
            break;
        case SceneObjectTypes::ADAMOPTIMIZER:
            m_adamEditorView->Render();
            break;
        case SceneObjectTypes::NERFDATASET:
            m_nerfDatasetInspectorView->Render();
            break;
        case SceneObjectTypes::NONE:
        default:
            ImGui::TextWrapped("Select an object in the object's list to edit its properties here.");
            break;
    }

    
    ImGui::End();
}