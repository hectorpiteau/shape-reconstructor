#pragma once

#include <string>
#include <vector>

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"
#include "../../../include/icons/IconsFontAwesome6.h"

#include "../../../model/ImageSet.hpp"
#include "../../../model/Image.hpp"
#include "../../../model/Texture2D.hpp"
#include "../../../model/Dataset/NeRFDataset.hpp"

#include "../../../interactors/NeRFInteractor.hpp"

class NeRFInspector
{
public:
    explicit NeRFInspector(NeRFInteractor *interactor) : m_interactor(interactor) {};

    NeRFInspector(const NeRFInspector &) = delete;

    ~NeRFInspector() = default;

    void Render()
    {
        if (m_interactor == nullptr)
        {
            ImGui::Text("Error: interactor is null.");
            return;
        }
        ImGui::SeparatorText(ICON_FA_INFO " NeRF Dataset - Information");
        
        ImGui::Text("ImageSet linked: Yes");
        ImGui::Text("Calibrations: ");
        ImGui::SameLine();
        if(m_interactor->IsCalibrationLoaded()){
            ImGui::TextColored(ImVec4(0.012f, 0.784f, 0.851f, 1.0f), "Loaded");
        }else{
            ImGui::TextColored(ImVec4(0.851f, 0.012f, 0.122f, 1.0f), "Not Loaded");
        }

        ImGui::Text("Cameras: ");
        ImGui::SameLine();
        if(m_interactor->AreCamerasGenerated()){
            ImGui::TextColored(ImVec4(0.012f, 0.784f, 0.851f, 1.0f), "Generated");
        }else{
            ImGui::TextColored(ImVec4(0.851f, 0.012f, 0.122f, 1.0f), "Not Generated");
        }

        ImGui::Spacing();
        ImGui::SeparatorText(ICON_FA_FILE " Source");

        const char *combo_preview = NeRFDatasetModesNames[m_comboBoxCurrent];
        static ImGuiComboFlags flags = 0;

        if (ImGui::BeginCombo("Dataset Type", combo_preview, flags))
        {
            for (size_t i = 0; i < NeRFDatasetModesNames.size(); ++i)
            {
                const bool is_selected = (m_comboBoxCurrent == i);
                if (ImGui::Selectable(NeRFDatasetModesNames[i], is_selected))
                {
                    if (m_comboBoxCurrent != i)
                    {
                        m_interactor->SetDatasetMode(i);
                    }
                    m_comboBoxCurrent = i;
                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::TextColored(ImVec4(8.0f, 0.0f, 8.0f, 1.0f), "- JSON path: ");
        ImGui::TextUnformatted(m_interactor->GetCurrentJsonPath().c_str());
        ImGui::Spacing();

        ImGui::SeparatorText(ICON_FA_GEARS " Actions");

        if (ImGui::Button("Load DataSet"))
        {
            m_interactor->LoadDataset();
        }

        ImGui::Spacing();
    };

private:
    NeRFInteractor *m_interactor;

    /** Combo */
    size_t m_comboBoxCurrent = 0;
};