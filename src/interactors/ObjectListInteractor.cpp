#include <memory>
#include "ObjectListInteractor.hpp"

#include "../controllers/Scene/Scene.hpp"
#include "../view/SceneObject/SceneObject.hpp"

ObjectListInteractor::ObjectListInteractor(std::shared_ptr<Scene>& scene) : m_scene(scene), m_selectedObject(nullptr)
{
    for(const auto& sceneObject : m_scene->GetSceneObjects()){
        
    }
}

void ObjectListInteractor::SetActive(bool active, int id)
{
    std::shared_ptr<SceneObject> tmp = m_scene->Get(id);
    if (tmp != nullptr)
        tmp->SetActive(active);
}

void ObjectListInteractor::Select(int id)
{
    std::shared_ptr<SceneObject> tmp = m_scene->Get(id);
    if (tmp != nullptr)
        m_selectedObject = tmp;
}

void ObjectListInteractor::Render(){
    
}