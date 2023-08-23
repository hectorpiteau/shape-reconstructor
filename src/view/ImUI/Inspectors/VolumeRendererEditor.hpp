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

class VolumeRendererEditor {
private:
    VolumeRendererInteractor *m_interactor;

    std::vector<std::shared_ptr<Camera>> m_comboBoxCameras = std::vector<std::shared_ptr<Camera>>();
    size_t m_comboBoxCurrent = 0;

public:
    explicit VolumeRendererEditor(VolumeRendererInteractor *interactor) : m_interactor(interactor) {
    }

    VolumeRendererEditor(const VolumeRendererEditor &) = delete;

    ~VolumeRendererEditor() = default;

    void Render() {
        vec2 renderZoneMinNDC = m_interactor->GetRenderingZoneMinNDC();
        vec2 renderZoneMaxNDC = m_interactor->GetRenderingZoneMaxNDC();

        vec2 renderZoneMinPixel = m_interactor->GetRenderingZoneMinPixel();
        vec2 renderZoneMaxPixel = m_interactor->GetRenderingZoneMaxPixel();

        bool showRenderingZone = m_interactor->IsRenderingZoneVisible();
        bool isRendering = m_interactor->IsRendering();

        m_comboBoxCameras = m_interactor->GetAvailableCameras();

        auto debugRayDesc = m_interactor->GetDebugRayDescriptor();

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
        ImGui::Checkbox("Render [v]", &isRendering);
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Checkbox("Show rendering zone on camera", &showRenderingZone);


        static ImGuiComboFlags flags = 0;

        /** Display image example. */
        if (!m_comboBoxCameras.empty()) {
            auto combo_preview = m_comboBoxCameras[m_comboBoxCurrent]->GetName().c_str();

            if (ImGui::BeginCombo("Target Camera", combo_preview, flags)) {
                for (size_t i = 0; i < m_comboBoxCameras.size(); ++i) {
                    const bool is_selected = (m_comboBoxCurrent == i);

                    if (ImGui::Selectable(m_comboBoxCameras[i]->GetName().c_str(), is_selected)) {
                        if (m_comboBoxCurrent != i) {
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

            if (showRenderingZone != m_interactor->IsRenderingZoneVisible()) {
                m_interactor->SetIsRenderingZoneVisible(showRenderingZone);
            }

            if (isRendering != m_interactor->IsRendering()) {
                m_interactor->SetIsRendering(isRendering);
            }

            ImGui::SeparatorText("Ray Samples Values");

            ImGui::TextWrapped("Place the cursor where you want then press : [ctrl] + [d]");

            ImGui::Spacing();
            ImGui::Spacing();
            static int drag_i = debugRayDesc->Host()->points;
            ImGui::DragInt("Show (0 -> 100)", &drag_i, 0.5f, 0, 100, "%d");

            ImVec2 scrolling_child_size = ImVec2(0, ImGui::GetFrameHeightWithSpacing() * 15 + 30);
            ImGui::BeginChild("scrolling", scrolling_child_size, true, ImGuiWindowFlags_HorizontalScrollbar);
            for(int i=0; i< drag_i; i++){
                auto name1 = std::string("Pos ##").append(std::to_string(i));
                auto name2 = std::string("Value ##").append(std::to_string(i));
                vec3 value1 = debugRayDesc->Host()->pointsWorldCoords[i];
                vec4 value2 = debugRayDesc->Host()->pointsSamples[i];
                ImGui::InputFloat3(name1.c_str(), &value1[0]);
                ImGui::InputFloat4(name2.c_str(), &value2[0]);
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
            }
            ImGui::EndChild();
        }
    }

};