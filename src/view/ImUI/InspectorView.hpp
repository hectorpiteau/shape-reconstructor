#ifndef INSPECTOR_VIEW_H
#define INSPECTOR_VIEW_H

#include <string>
#include <memory>
#include <vector>

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"

#include "../../../include/icons/IconsFontAwesome6.h"

#include "Inspectors/ImageSetInspector.hpp"
#include "Inspectors/CameraInspector.hpp"
#include "Inspectors/NeRFInspector.hpp"

#include "../../interactors/SceneObjectInteractor.hpp"

class SceneObjectInteractor;

/**
 * @brief ImGui Window containing an object's inspector. 
 * 
 */
class InspectorView {
public:
    InspectorView(SceneObjectInteractor* interactor);
    InspectorView(const InspectorView&) = delete;
    ~InspectorView();

    /**
     * @brief Set the inspector's interactor.
     * Used as the intermediate to other interactors.
     * 
     * @param interactor : An interactor used to interact with 
     */
    void SetInteractor(std::shared_ptr<SceneObjectInteractor>& interactor);

    /**
     * @brief Set the Selected object informations that will allow to display
     * a view that matchs the object's editable properties.
     * 
     * @param type : The type of the selected SceneObject.
     * @param interactor : The interactor that is used to interact with the current SceneObject.
     */
    void SetSelected(enum SceneObjectTypes type);

    void Render();

private:

    /** Current interactor to be used. */
    SceneObjectInteractor* m_interactor;

    /** Inspectors */
    ImageSetInspector* m_imageSetInspectorView;
    CameraInspector* m_cameraInspectorView;
    NeRFInspector* m_nerfDatasetInspectorView;

    std::string m_selectedName;
    int m_selectedId;
    enum SceneObjectTypes m_selectedType;
};


#endif //INSPECTOR_VIEW_H