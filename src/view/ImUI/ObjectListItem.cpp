#include <string>
#include <memory>
#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"
#include "../../interactors/ObjectListInteractor.hpp"

#include "ObjectListItem.hpp"

#define ID_XX "##"

ObjectListItem::ObjectListItem(const std::string &name, int id, bool checked, std::vector<std::shared_ptr<ObjectListItem>>& children, ObjectListInteractor* interactor) : 
m_name(name), 
m_checked(checked), 
m_id(id), 
m_locked(false),
m_interactor(interactor), 
m_children(children) 
{
}

void ObjectListItem::SetChecked(bool checked)
{
    m_checked = checked;
}

void ObjectListItem::SetLocked(bool locked){
    m_locked = locked;
}

void ObjectListItem::Render()
{
    bool islok = m_locked;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();

    std::string checkbox_id = std::string(ID_XX).append(m_name);

    if(ImGui::Checkbox(checkbox_id.c_str(), &m_checked)){
        std::cout << "Checkbox changed, id: " << m_id << std::endl;
        m_interactor->SetActive(m_checked, m_id);
    }

    ImGui::TableNextColumn();

    if(islok){
        ImGui::Text(ICON_FA_CARET_RIGHT " ");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.0f, 0.6f, 0.6f));    
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.05f, 0.8f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.0f, 0.8f, 0.8f));
    }

    if(ImGui::Button(m_name.c_str())){
        std::cout << "Button clicked, id: " << m_id << std::endl;
        m_interactor->Select(m_id);
    }

    if(islok){
        ImGui::PopStyleColor(3);
    }

    ImGui::TableNextColumn();


    for(const auto& child : m_children){
        child->Render();
    }
}
