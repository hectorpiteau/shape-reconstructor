#ifndef OBJECT_LIST_VIEW_H
#define OBJECT_LIST_VIEW_H

#include <string>
#include <memory>
#include <vector>
#include "../../interactors/ObjectListInteractor.hpp"
#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"

#include "../../../include/icons/IconsFontAwesome6.h"

#include "ObjectListItem.hpp"

/**
 * @brief ImGui Window containing a list of all objects. 
 * Display also an Inspector for each object's type.
 */
class ObjectListView {
public:
    ObjectListView(std::shared_ptr<ObjectListInteractor> interactor);

    void AddItem(std::shared_ptr<ObjectListItem> item);

    void SetSelected(int id, const std::string& name);

    void Render();

private:
    std::vector<std::shared_ptr<ObjectListItem>> m_items;

    std::shared_ptr<ObjectListInteractor> m_interactor;

    std::string m_selectedName;
    int m_selectedId;
};


#endif //OBJECT_LIST_VIEW_H