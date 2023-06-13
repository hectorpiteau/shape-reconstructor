#pragma once
#include <memory>
#include <glm/glm.hpp>

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"

#include "../../../include/icons/IconsFontAwesome6.h"

#include "../../../interactors/CameraInteractor.hpp"
#include "../../../model/Camera/Camera.hpp"

using namespace glm;

class CameraInspector
{
public:
    explicit CameraInspector(CameraInteractor *interactor) : m_interactor(interactor){};
    CameraInspector(const CameraInspector &) = delete;

    ~CameraInspector()= default;

    void Render()
    {
        if(m_interactor == nullptr) return;
        if(m_isExternal){ 
            ImGui::Begin("External Camera Editor", &m_isOpened);
        }

        m_pos = m_interactor->GetPosition();
        m_target = m_interactor->GetTarget();
        m_right = m_interactor->GetRight();
        m_up = m_interactor->GetUp();
        m_near = m_interactor->GetNear();
        m_far = m_interactor->GetFar();

        m_intrinsicT = glm::transpose(m_interactor->GetIntrinsic());
        m_extrinsicT = glm::transpose(m_interactor->GetExtrinsic());

        bool dispCenterLine = m_interactor->IsCenterLineVisible();
        bool dispFrustLines = m_interactor->ShowFrustumLines();
        bool dispImagePlane = m_interactor->ShowImagePlane();

        float centerLineLength = m_interactor->GetCenterLineLength();
        
        /** Camera id. */
        ImGui::SeparatorText("Camera");
        if (m_interactor->GetCamera() == nullptr)
        {
            ImGui::Text("Camera is null");
            return;
        }

        ImGui::TextUnformatted(m_interactor->GetCamera()->filename.c_str());

        ImGui::Checkbox("Display Image Plane", &dispImagePlane);
        ImGui::Checkbox("Display Frustum", &dispFrustLines);

        ImGui::Separator();
        ImGui::Spacing();
        
        ImGui::Text("Instrinsic Matrix: ");
        ImGui::BeginDisabled();
        ImGui::InputFloat4("##CameraIntrinsic0", &m_intrinsicT[0][0]);
        ImGui::InputFloat4("##CameraIntrinsic1", &m_intrinsicT[1][0]);
        ImGui::InputFloat4("##CameraIntrinsic2", &m_intrinsicT[2][0]);
        ImGui::InputFloat4("##CameraIntrinsic3", &m_intrinsicT[3][0]);
        ImGui::EndDisabled();

        ImGui::Text("Extrinsic Matrix: ");
        ImGui::BeginDisabled();
        ImGui::InputFloat4("##CameraExtrinsic0", &m_extrinsicT[0][0]);
        ImGui::InputFloat4("##CameraExtrinsic1", &m_extrinsicT[1][0]);
        ImGui::InputFloat4("##CameraExtrinsic2", &m_extrinsicT[2][0]);
        ImGui::InputFloat4("##CameraExtrinsic3", &m_extrinsicT[3][0]);
        ImGui::EndDisabled();

        
        ImGui::SeparatorText("Properties");

        ImGui::InputFloat3("Pos", &m_pos[0]);
        ImGui::InputFloat3("Target", &m_target[0]);
        ImGui::InputFloat3("Right", &m_right[0]);
        ImGui::InputFloat3("Up", &m_up[0]);
        ImGui::Separator();
        ImGui::InputFloat("Near", &m_near);
        ImGui::InputFloat("Far", &m_far);
        ImGui::Separator();
        
        ImGui::Checkbox("Display Center Line", &dispCenterLine);
        ImGui::InputFloat("Center Line Length", &centerLineLength);
        


        /** Update values in model if changed. */

        if (all(equal(m_interactor->GetPosition(), m_pos)) == false)
        {
            m_interactor->SetPosition(m_pos);
        }

        if (all(equal(m_interactor->GetTarget(), m_target)) == false)
        {
            m_interactor->SetTarget(m_target);
        }

        if (all(equal(m_interactor->GetRight(), m_right)) == false)
        {
            m_interactor->SetRight(m_right);
        }

        if (all(equal(m_interactor->GetUp(), m_up)) == false)
        {
            m_interactor->SetUp(m_up);
        }

        if(m_isExternal){ 
            ImGui::End();
        }

        if(m_interactor->ShowImagePlane() != dispImagePlane)
            m_interactor->SetShowImagePlane(dispImagePlane);

        if(m_interactor->ShowFrustumLines() != dispFrustLines)
            m_interactor->SetShowFrustumLines(dispFrustLines);

        if(m_interactor->IsCenterLineVisible() != dispCenterLine)
            m_interactor->SetIsCenterLineVisible(dispCenterLine);

        if(centerLineLength != m_interactor->GetCenterLineLength())
            m_interactor->SetCenterLineLength(centerLineLength);
    };

    void SetIsOpened(bool opened) {m_isOpened = opened;}

    bool IsOpened(){
        return m_isOpened;
    }

    bool IsExternal(){
        return m_isExternal;
    }

    void SetAsExternal(){
        m_isExternal = true;
    }
private:
    CameraInteractor *m_interactor;

    mat4 m_intrinsicT{};
    mat4 m_extrinsicT{};
    
    vec3 m_pos{};
    vec3 m_target{};
    vec3 m_right{};
    vec3 m_up{};
    float m_near{}, m_far{};

    bool m_isOpened = false;
    bool m_isExternal = false;
};