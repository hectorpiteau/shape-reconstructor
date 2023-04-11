#include <memory>
#include <iostream>
#include <string>

#include "../controllers/Scene/Scene.hpp"
#include "../view/SceneObject/SceneObject.hpp"
#include "../view/ImUI/ObjectListItem.hpp"

#include "ObjectListInteractor.hpp"

ObjectListInteractor::ObjectListInteractor(std::shared_ptr<Scene> &scene, std::shared_ptr<ObjectListView> listView, std::shared_ptr<SceneObjectInteractor>& sceneObjectInteractor)
    : m_scene(scene), m_objectListView(listView), m_selectedObject(nullptr), m_sceneObjectInteractor(sceneObjectInteractor)
{
    listView->SetInteractor(this);

    for (const auto &sceneObject : m_scene->GetSceneObjects())
    {
        if(sceneObject != nullptr){
            m_objectListView->AddItem(
                std::make_shared<ObjectListItem>(sceneObject->GetName(), sceneObject->GetID(), sceneObject->IsActive(), this)
            );
        }
    }
}

ObjectListInteractor::~ObjectListInteractor(){
}

void ObjectListInteractor::SetActive(bool active, int id)
{
    std::shared_ptr<SceneObject> tmp = m_scene->Get(id);
    if (tmp != nullptr)
        tmp->SetActive(active);
    else {
        std::cout << "err" << std::endl;
    }
}

void ObjectListInteractor::Select(int id)
{
    std::shared_ptr<SceneObject> tmp = m_scene->Get(id);
    if (tmp != nullptr){
        std::cout << "ObjectListInteractor::Select " << std::to_string(id) << std::endl;
        m_selectedObject = tmp;
        /** Update the sceneObject's interactor's selected SceneObject. */
        m_sceneObjectInteractor->SetSelectedSceneObject(tmp);
    }


}

void ObjectListInteractor::Render()
{
    m_scene->RenderAll();
    m_objectListView->Render();
}