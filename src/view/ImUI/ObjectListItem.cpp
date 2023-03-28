#include <string>
#include <memory>
#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"
#include "../../interactors/ObjectListInteractor.hpp"

#include "ObjectListItem.hpp"

ObjectListItem::ObjectListItem(const std::string &name, int id, std::shared_ptr<ObjectListInteractor> interactor) : m_name(name), m_id(id), m_interactor(interactor)
{
}

void ObjectListItem::SetChecked(bool checked)
{
    m_checked = checked;
}

void ObjectListItem::Render()
{
    bool tmp, tmp2;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::Checkbox("", &tmp);
    ImGui::TableNextColumn();
    tmp2 = ImGui::Button(m_name.c_str());

    if (tmp != m_checked)
    {
        m_checked = tmp;
        m_interactor->SetActive(tmp, m_id);
    }

    if (tmp2)
    {
        m_interactor->Select(m_id);
    }
}
