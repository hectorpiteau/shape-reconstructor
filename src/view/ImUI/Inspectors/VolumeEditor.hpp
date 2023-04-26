/*
Author: Hector Piteau (hector.piteau@gmail.com)
VolumeEditor.hpp (c) 2023
Desc: Editor panel of the Volume3D scene object.
Created:  2023-04-21T12:12:19.078Z
Modified: 2023-04-25T14:08:01.696Z
*/
#pragma once
#include <glm/glm.hpp>

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"
#include "../../../include/icons/IconsFontAwesome6.h"

#include "../../../interactors/Volume3DInteractor.hpp"

using namespace glm;

class VolumeEditor
{
private:
    Volume3DInteractor *m_interactor;

    vec3 bboxMinimum = vec3(0.0f);
    vec3 bboxMaximum = vec3(0.0f);

public:
    VolumeEditor(Volume3DInteractor *interactor) : m_interactor(interactor){

    }

    VolumeEditor(const VolumeEditor& ) = delete;

    ~VolumeEditor() {

    }

    void Render(){
        
        bboxMinimum = m_interactor->GetBboxMin();
        bboxMaximum = m_interactor->GetBboxMax();

        const vec3* bbox = m_interactor->GetBBox();
        vec3 bboxe[8] = {
            bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],bbox[5],bbox[6],bbox[7],
        };
        
        ImGui::SeparatorText(ICON_FA_INFO " Volume3D - Informations");
        
        ImGui::InputFloat3("BBox Minimum", &bboxMinimum[0]);
        ImGui::InputFloat3("BBox Maximum", &bboxMaximum[0]);

        ImGui::Separator();
        
        ImGui::InputFloat3("BBox 0", &bboxe[0][0]);
        ImGui::InputFloat3("BBox 1", &bboxe[1][0]);
        ImGui::InputFloat3("BBox 2", &bboxe[2][0]);
        ImGui::InputFloat3("BBox 3", &bboxe[3][0]);
        ImGui::InputFloat3("BBox 4", &bboxe[4][0]);
        ImGui::InputFloat3("BBox 5", &bboxe[5][0]);
        ImGui::InputFloat3("BBox 6", &bboxe[6][0]);
        ImGui::InputFloat3("BBox 7", &bboxe[7][0]);
        // ImGui::InputFloat3("BBox Maximum", &Rendering);



         
    }
};