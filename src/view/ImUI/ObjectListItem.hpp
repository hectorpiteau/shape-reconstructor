#pragma once

#include <string>
#include <memory>

#include "../../interactors/ObjectListInteractor.hpp"

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"

class ObjectListInteractor;

class ObjectListItem {
public:
    
    ObjectListItem(const std::string &name, int id, bool checked, std::vector<std::shared_ptr<ObjectListItem>>& children, ObjectListInteractor* interactor);

    void SetChecked(bool checked);

    void SetLocked(bool locked);

    void Render();

    //TODO: move from here.
    const std::string& m_name;
private:

    bool m_checked;

    int m_id;

    ObjectListInteractor* m_interactor;

    bool m_locked;

    std::vector<std::shared_ptr<ObjectListItem>> m_children;
};