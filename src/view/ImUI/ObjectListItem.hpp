#ifndef OBJECT_LIST_ITEM_H
#define OBJECT_LIST_ITEM_H

#include <string>
#include <memory>
#include "../../interactors/ObjectListInteractor.hpp"
#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"

class ObjectListItem {
public:
    
    ObjectListItem(const std::string &name, int id, std::shared_ptr<ObjectListInteractor> interactor);

    void SetChecked(bool checked);

    void Render();

private:
    const std::string& m_name;

    bool m_checked;

    int m_id;

    std::shared_ptr<ObjectListInteractor> m_interactor;
};


#endif //OBJECT_LIST_ITEM_H