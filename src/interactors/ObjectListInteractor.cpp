#include <memory>
#include <iostream>
#include <string>

#include "../controllers/Scene/Scene.hpp"
#include "../view/SceneObject/SceneObject.hpp"
#include "../view/ImUI/ObjectListItem.hpp"

#include "ObjectListInteractor.hpp"

/**
 * @brief Helper function to recursively create ObjectListItems for the SceneObject given
 * and it's dependencies at once. 
 * 
 * @param sceneObject : The SceneObject to create its corresponding ObjectListItem.
 * @param interactor : The interactor that the items are going to communicate with. 
 * @return std::shared_ptr<ObjectListItem> : A shared pointer of the ObjectListItem that correspond to the SceneObject given.
 */
std::shared_ptr<ObjectListItem> CreateObjectListItem(const std::shared_ptr<SceneObject> &sceneObject, ObjectListInteractor* interactor){
    std::vector<std::shared_ptr<ObjectListItem>> result =  std::vector<std::shared_ptr<ObjectListItem>>();

    /** Loop trough children with the same function to generate an ObjectListItem per child. */
    for(const auto& child: sceneObject->GetChildren()){
        if(!child->IsVisibleInList()) continue;
        std::shared_ptr<ObjectListItem> childItem = CreateObjectListItem(child, interactor);
        childItem->SetLocked(true);
        result.push_back(childItem);
    }

    return std::make_shared<ObjectListItem>(sceneObject->GetName(), sceneObject->GetID(), sceneObject->IsActive(), result, interactor);
}

ObjectListInteractor::ObjectListInteractor(Scene* scene, std::shared_ptr<ObjectListView> listView, std::shared_ptr<SceneObjectInteractor>& sceneObjectInteractor) : 
m_scene(scene), 
m_objectListView(listView), 
m_sceneObjectInteractor(sceneObjectInteractor),
m_selectedObject(nullptr) 
{
    listView->SetInteractor(this);

    UpdateList();
}

void ObjectListInteractor::UpdateList(){
    if(m_objectListView == nullptr) return;
    
    m_objectListView->EmptyList();
    
    for (const auto &sceneObject : m_scene->GetSceneObjects())
    {
        if(sceneObject == nullptr || sceneObject->IsChild()) continue;
        m_objectListView->AddItem(CreateObjectListItem(sceneObject, this));
    }
}


ObjectListInteractor::~ObjectListInteractor(){}

void ObjectListInteractor::SetActive(bool active, int id)
{
    std::shared_ptr<SceneObject> tmp = m_scene->Get(id);
    if (tmp != nullptr)
        tmp->SetActive(active);
    else {
        std::cout << "Error SceneObject does not exist." << std::endl;
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

void ObjectListInteractor::SelectMainCamera(){
    Select(m_scene->GetDefaultCam()->GetID());
}

void ObjectListInteractor::Render()
{
    m_scene->RenderAll();
    m_objectListView->Render();
}