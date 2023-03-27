
#include "../../view/SceneObject/SceneObject.hpp"
#include "../../utils/SceneSettings.hpp"
#include "../../model/UniqId.hpp"
#include "Scene.hpp"
#include <vector>
#include <memory>


Scene::Scene(std::shared_ptr<SceneSettings> sceneSettings) : m_sceneSettings(sceneSettings){
    m_uniqIdManager = std::make_unique<UniqId>(128);
}

Scene::~Scene(){
    //do nothing for now
}

int Scene::Add(std::shared_ptr<SceneObject> object){

}

void Scene::Remove(int id){

}


void Scene::RenderAll()
{
    for(auto &obj : m_objects){
        /** Render only active objects. (The checkbox in the scene object's lists). */
        if(obj->IsActive()){
            obj->Render(m_activeCamera->GetProjectionMatrix(), m_activeCamera->GetViewMatrix(), m_sceneSettings);
        }
    }
}
