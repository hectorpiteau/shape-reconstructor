#ifndef OBJECT_LIST_ITEM_H
#define OBJECT_LIST_ITEM_H

#include <string>
#include <memory>
#include "../../interactors/ObjectListInteractor.hpp"
#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"

class ObjectListInteractor;

class ObjectListItem {
public:
    
    ObjectListItem(const std::string &name, int id, bool checked, ObjectListInteractor* interactor);

    void SetChecked(bool checked);

    void Render();

    const std::string& m_name;
private:

    bool m_checked;

    int m_id;

    ObjectListInteractor* m_interactor;
};


#endif //OBJECT_LIST_ITEM_H