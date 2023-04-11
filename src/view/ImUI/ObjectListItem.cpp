#include <string>
#include <memory>
#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"
#include "../../interactors/ObjectListInteractor.hpp"

#include "ObjectListItem.hpp"

#define ID_XX "##"

ObjectListItem::ObjectListItem(const std::string &name, int id, bool checked, ObjectListInteractor* interactor) : m_name(name), m_id(id), m_checked(checked), m_interactor(interactor)
{
}

void ObjectListItem::SetChecked(bool checked)
{
    m_checked = checked;
}

void ObjectListItem::Render()
{
    ImGui::TableNextRow();
    ImGui::TableNextColumn();

    std::string checkbox_id = std::string(ID_XX).append(m_name);

    if(ImGui::Checkbox(checkbox_id.c_str(), &m_checked)){
        std::cout << "Checkbox changed, id: " << m_id << std::endl;
        m_interactor->SetActive(m_checked, m_id);
    }

    ImGui::TableNextColumn();
    
    if(ImGui::Button(m_name.c_str())){
        std::cout << "Button clicked, id: " << m_id << std::endl;
        m_interactor->Select(m_id);
    }
    ImGui::TableNextColumn();
}
