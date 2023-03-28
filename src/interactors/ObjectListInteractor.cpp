#include <memory>
#include "ObjectListInteractor.hpp"

#include "../controllers/Scene/Scene.hpp"
#include "../view/SceneObject/SceneObject.hpp"
#include "../view/ImUI/ObjectListItem.hpp"

ObjectListInteractor::ObjectListInteractor(std::shared_ptr<Scene> &scene, std::shared_ptr<ObjectListView> listView)
    : m_scene(scene), m_objectListView(listView), m_selectedObject(nullptr)
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
    if (tmp != nullptr)
        m_selectedObject = tmp;
}

void ObjectListInteractor::Render()
{
    m_scene->RenderAll();
    m_objectListView->Render();
}