#pragma once

#include <string>
#include <vector>

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"

#include "../../../include/icons/IconsFontAwesome6.h"

#include "../../../interactors/CameraSetInteractor.hpp"

#include "CameraInspector.hpp"

class CameraSetInspector
{
public:
    CameraSetInspector(CameraSetInteractor *interactor) : m_interactor(interactor){

        m_cameraInteractor = new CameraInteractor();
        m_cameraInspector = new CameraInspector(m_cameraInteractor);
        m_cameraInspector->SetAsExternal();
    };

    CameraSetInspector(const CameraSetInspector &) = delete;

    ~CameraSetInspector(){
        delete m_cameraInteractor;
        delete m_cameraInspector;

    };

    void Render()
    {
        if (m_interactor == nullptr)
        {
            ImGui::Text("Error: interactor is null.");
            return;
        }
        ImGui::SeparatorText(ICON_FA_INFO " CameraSet - Informations");

        ImGui::Text("ImageSet linked: Yes");
        // ImGui::Text("Calibrations: ");
        // ImGui::SameLine();
        // if (m_interactor->IsCalibrationLoaded())
        // {
        //     ImGui::TextColored(ImVec4(0.012f, 0.784f, 0.851f, 1.0f), "Loaded");
        // }
        // else
        // {
        //     ImGui::TextColored(ImVec4(0.851f, 0.012f, 0.122f, 1.0f), "Not Loaded");
        // }

        if (ImGui::Button("Generate cameras"))
        {
        }
        ImGui::SameLine();
        if (m_interactor->AreCamerasGenerated())
            ImGui::TextColored(ImVec4(0.012f, 0.784f, 0.851f, 1.0f), "Generated");
        else
            ImGui::TextColored(ImVec4(0.851f, 0.012f, 0.122f, 1.0f), "Not Generated");

        ImGui::Checkbox("Linked to an ImageSet", &m_linkedToImageSet);

        static ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

        static const float TEXT_BASE_HEIGHT = ImGui::GetTextLineHeightWithSpacing();
        ImVec2 outer_size = ImVec2(0.0f, TEXT_BASE_HEIGHT * 8);
        if (ImGui::BeginTable("table_scrolly", 2, flags, outer_size))
        {
            ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
            ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_None);
            ImGui::TableSetupColumn("Camera", ImGuiTableColumnFlags_None);
            ImGui::TableHeadersRow();

            for(auto cam : m_interactor->GetCameras()){
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%d", cam->GetID());
                ImGui::TableSetColumnIndex(1);
                if(ImGui::Button(cam->GetName().c_str())){
                    std::cout << "Click on camera editor: " << cam->GetName() << std::endl;
                    m_cameraInteractor->SetCamera(cam);
                    m_cameraInspector->SetIsOpened(true);
                }
            }
            ImGui::EndTable();
        }

        ImGui::Spacing();
        ImGui::SeparatorText(ICON_FA_GEARS " Actions");

        if(m_cameraInspector->IsOpened()){
            m_cameraInspector->Render();
        }
    }



    private:
        CameraSetInteractor *m_interactor;
        
        bool m_linkedToImageSet = true;

        /** External camera editor. */
        std::shared_ptr<Camera> m_camera;
        CameraInspector *m_cameraInspector;
        CameraInteractor *m_cameraInteractor;

        /** Combo */
        int m_comboBoxCurrent = 0;
    
        

    };