#pragma once
#include <memory>

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"

#include "../../../include/icons/IconsFontAwesome6.h"


#include "../../../interactors/CameraInteractor.hpp"
#include "../../../model/Camera/Camera.hpp"

class CameraInspector {
public:
    CameraInspector(){ m_camera = nullptr; };
    CameraInspector(const CameraInspector&) = delete;

    ~CameraInspector(){};

    void SetCamera(std::shared_ptr<Camera>& camera){
        m_camera = camera;
    }

    void Render(CameraInteractor* interactor){
        /** Camera id. */
        ImGui::SeparatorText("Camera");
        if(m_camera == nullptr){
            ImGui::Text("Camera is null");
        }else{
            
        }
    };

private:
    std::shared_ptr<Camera> m_camera;

};