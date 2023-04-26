/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeRendererEditor.hpp (c) 2023
Desc: VolumeRendererEditor
Created:  2023-04-21T12:47:52.699Z
Modified: 2023-04-26T09:24:24.924Z
*/
#pragma once
#include <glm/glm.hpp>

#include "../../../interactors/VolumeRendererInteractor.hpp"

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"
#include "../../../include/icons/IconsFontAwesome6.h"

using namespace glm;

class VolumeRendererEditor
{
private:
    VolumeRendererInteractor *m_interactor;

    std::vector<std::shared_ptr<Camera>> m_comboBoxCameras = std::vector<std::shared_ptr<Camera>>();
    size_t m_comboBoxCurrent = 0;

public:
    VolumeRendererEditor(VolumeRendererInteractor *interactor) : m_interactor(interactor)
    {
    }

    VolumeRendererEditor(const VolumeRendererEditor &) = delete;

    ~VolumeRendererEditor()
    {
    }

    void Render()
    {
        vec2 renderZoneMinNDC = m_interactor->GetRenderingZoneMinNDC();
        vec2 renderZoneMaxNDC = m_interactor->GetRenderingZoneMaxNDC();

        vec2 renderZoneMinPixel = m_interactor->GetRenderingZoneMinPixel();
        vec2 renderZoneMaxPixel = m_interactor->GetRenderingZoneMaxPixel();

        bool showRenderingZone = m_interactor->IsRenderingZoneVisible();

        m_comboBoxCameras = m_interactor->GetAvailableCameras();

        ImGui::SeparatorText("VolumeRenderer");

        ImGui::BeginDisabled();
        ImGui::Text("Rendering zone NDC: ");
        ImGui::InputFloat2("Minimum", &renderZoneMinNDC[0]);
        ImGui::InputFloat2("Maximum", &renderZoneMaxNDC[0]);
        ImGui::Text("Rendering zone pixels: ");
        ImGui::InputFloat2("Minimum", &renderZoneMinPixel[0]);
        ImGui::InputFloat2("Maximum", &renderZoneMaxPixel[0]);
        ImGui::EndDisabled();
        ImGui::Separator();

        ImGui::Spacing();
        ImGui::Checkbox("Show rendering zone on camera", &showRenderingZone);

        static ImGuiComboFlags flags = 0;

        /** Display image example. */
        if (m_comboBoxCameras.size() > 0)
        {
            auto combo_preview = m_comboBoxCameras[m_comboBoxCurrent]->GetName().c_str();

            if (ImGui::BeginCombo("Target Camera", combo_preview, flags))
            {
                for (int i = 0; i < m_comboBoxCameras.size(); ++i)
                {

                    const bool is_selected = (m_comboBoxCurrent == i);

                    if (ImGui::Selectable(m_comboBoxCameras[i]->GetName().c_str(), is_selected))
                    {
                        if (m_comboBoxCurrent != i)
                        {
                            m_interactor->SetTargetCamera(m_comboBoxCameras[i]);
                        }
                        m_comboBoxCurrent = i;
                    }

                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            if (showRenderingZone != m_interactor->IsRenderingZoneVisible())
            {
                m_interactor->SetIsRenderingZoneVisible(showRenderingZone);
            }

            ImGui::Separator();
        }
    }

};