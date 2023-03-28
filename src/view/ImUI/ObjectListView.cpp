#include <string>
#include <memory>
#include <vector>

#include "ObjectListView.hpp"

#include "../../interactors/ObjectListInteractor.hpp"
#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"
#include "../../../include/icons/IconsFontAwesome6.h"

#include "ObjectListItem.hpp"

ObjectListView::ObjectListView(std::shared_ptr<ObjectListInteractor> interactor) : m_interactor(interactor), m_items(10)
{
}

void ObjectListView::AddItem(std::shared_ptr<ObjectListItem> item)
{
    m_items.push_back(item);
}

void ObjectListView::SetSelected(int id, const std::string &name)
{
    m_selectedId = id;
    m_selectedName = name;
}

void ObjectListView::Render()
{
    ImGui::Begin("Objects");

    ImGui::SeparatorText("Objects in Scene");

    static ImGuiTableFlags flags = ImGuiTableFlags_BordersV | ImGuiTableFlags_BordersOuterH | ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoBordersInBody;

    if (ImGui::BeginTable("3ways", 3, flags))
    {
        // The first column will use the default _WidthStretch when ScrollX is Off and _WidthFixed when ScrollX is On
        ImGui::TableSetupColumn("Active", ImGuiTableColumnFlags_NoHide);
        ImGui::TableSetupColumn("Object Name", ImGuiTableColumnFlags_NoHide);
        ImGui::TableSetupColumn("Settings", ImGuiTableColumnFlags_NoHide);
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        bool test = true;
        ImGui::BeginDisabled();
        ImGui::Checkbox("", &test);
        ImGui::EndDisabled();
        ImGui::TableNextColumn();
        ImGui::Text(std::string(ICON_FA_CAMERA " Camera 0 (main)").c_str());

        for (auto &item : m_items)
        {
            item->Render();
        }

        ImGui::EndTable();
    }

    ImGui::SeparatorText("Inspector");

    ImGui::End();
}
