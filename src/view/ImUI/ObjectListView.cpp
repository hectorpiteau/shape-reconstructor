#include <string>
#include <memory>
#include <vector>

#include "ObjectListItem.hpp"
#include "ObjectListView.hpp"

#include "../../interactors/ObjectListInteractor.hpp"
#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"
#include "../../../include/icons/IconsFontAwesome6.h"


ObjectListView::ObjectListView()
{
}

void ObjectListView::SetInteractor(ObjectListInteractor* interactor){
    m_interactor = interactor;
}

void ObjectListView::AddItem(std::shared_ptr<ObjectListItem> item)
{
    m_items.push_back(item);
}

void ObjectListView::EmptyList(){
    m_items.clear();
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

    if(ImGui::Button(ICON_FA_ROTATE_RIGHT " Update")){
        m_interactor->UpdateList();
    }

    static ImGuiTableFlags flags = ImGuiTableFlags_BordersV | ImGuiTableFlags_BordersOuterH | ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoBordersInBody;

    if (ImGui::BeginTable("3ways", 3, flags))
    {
        // The first column will use the default _WidthStretch when ScrollX is Off and _WidthFixed when ScrollX is On
        ImGui::TableSetupColumn("Active", ImGuiTableColumnFlags_NoHide);
        ImGui::TableSetupColumn("Object Name", ImGuiTableColumnFlags_NoHide);
        ImGui::TableSetupColumn(" ", ImGuiTableColumnFlags_NoHide);
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        bool test = true;
        ImGui::BeginDisabled();
        ImGui::Checkbox("", &test);
        ImGui::EndDisabled();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(std::string(ICON_FA_CAMERA " Camera 0 (main)").c_str());
        ImGui::TableNextColumn();
        if(ImGui::Button(std::string(" " ICON_FA_GEAR " ").c_str())){
            m_interactor->SelectMainCamera();
        }

        for (size_t i=0; i<m_items.size(); ++i)
        {
            m_items[i]->Render();
        }

        ImGui::EndTable();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        static int current = 0;
        ImGui::Combo("##AddSceneObjectComboBox", &current, SceneObjectNames[0]);
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(2.0f / 7.0f, 0.6f, 0.6f));    
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(2.0f / 7.0f, 0.7f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(2.0f / 7.0f, 0.8f, 0.8f));

        if(ImGui::Button("Add Object")){
            // add the selected SceneObject.
        }

        ImGui::PopStyleColor(3);
    }
    ImGui::End();
}
