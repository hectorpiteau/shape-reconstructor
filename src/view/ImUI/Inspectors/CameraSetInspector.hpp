#pragma once

#include <string>
#include <vector>

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"

#include "../../../include/icons/IconsFontAwesome6.h"

#include "../../../interactors/CameraSetInteractor.hpp"

class CameraSetInspector
{
public:
    CameraSetInspector(CameraSetInteractor *interactor) : m_interactor(interactor){};

    CameraSetInspector(const CameraSetInspector &) = delete;

    ~CameraSetInspector(){};

    void Render()
    {
        if (m_interactor == nullptr)
        {
            ImGui::Text("Error: interactor is null.");
            return;
        }
        ImGui::SeparatorText(ICON_FA_INFO " CameraSet - Informations");

        ImGui::Text("ImageSet linked: Yes");
        ImGui::Text("Calibrations: ");
        ImGui::SameLine();
        if (m_interactor->IsCalibrationLoaded())
        {
            ImGui::TextColored(ImVec4(0.012f, 0.784f, 0.851f, 1.0f), "Loaded");
        }
        else
        {
            ImGui::TextColored(ImVec4(0.851f, 0.012f, 0.122f, 1.0f), "Not Loaded");
        }

        if (ImGui::Button("Generate cameras"))
        {
        }
        ImGui::SameLine();
        if (m_interactor->AreCamerasGenerated())
            ImGui::TextColored(ImVec4(0.012f, 0.784f, 0.851f, 1.0f), "Generated");
        else
            ImGui::TextColored(ImVec4(0.851f, 0.012f, 0.122f, 1.0f), "Not Generated");

        ImGui::Checkbox("Linked to an ImageSet");

        if (ImGui::BeginTable("3ways", 3, flags))
        {
            // The first column will use the default _WidthStretch when ScrollX is Off and _WidthFixed when ScrollX is On
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoHide);
            ImGui::TableSetupColumn("Image", ImGuiTableColumnFlags_NoHide);
            ImGui::TableSetupColumn("Object", ImGuiTableColumnFlags_NoHide);
            ImGui::TableHeadersRow();

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            bool test = true;
            ImGui::BeginDisabled();
            ImGui::Checkbox("", &test);
            ImGui::EndDisabled();
            ImGui::TableNextColumn();
            ImGui::Text(std::string(ICON_FA_CAMERA " Camera 0 (main)").c_str());
            ImGui::TableNextColumn();
            ImGui::Button(std::string(" " ICON_FA_GEAR " ").c_str());

            for (int i = 0; i < m_items.size(); ++i)
            {
                m_items[i]->Render();
            }

            ImGui::EndTable();

            ImGui::Spacing();

            ImGui::SeparatorText(ICON_FA_GEARS " Actions");

            if (ImGui::Button("Load Calibrations"))
            {
            }

            if (ImGui::Button("Generate cameras"))
            {
            }

            ImGui::Spacing();
        };

    private:
        CameraSetInteractor *m_interactor;

        /** Combo */
        int m_comboBoxCurrent = 0;
    };