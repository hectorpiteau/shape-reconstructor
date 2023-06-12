#pragma once 

#include <string>
#include <memory>
#include <vector>

#include "ObjectListItem.hpp"

#include "../../interactors/ObjectListInteractor.hpp"
#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"

#include "../../../include/icons/IconsFontAwesome6.h"


class ObjectListInteractor;
class ObjectListItem;

/**
 * @brief ImGui Window containing a list of all objects. 
 */
class ObjectListView {
public:
    ObjectListView();
    
    void AddItem(std::shared_ptr<ObjectListItem> item);

    void SetSelected(int id, const std::string& name);

    void SetInteractor(ObjectListInteractor* interactor);

    void Render();

    void EmptyList();

    

private:
    std::vector<std::shared_ptr<ObjectListItem>> m_items;

    ObjectListInteractor* m_interactor{};

    std::string m_selectedName;
    int m_selectedId{};
};