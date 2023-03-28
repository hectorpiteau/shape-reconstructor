#include <vector>
#include <memory>

#include "Scene.hpp"

#include "../../view/SceneObject/SceneObject.hpp"
#include "../../utils/SceneSettings.hpp"
#include "../../model/UniqId.hpp"
#include "../../view/Volume3D.hpp"
#include "../../view/LineGrid.hpp"


Scene::Scene(std::shared_ptr<SceneSettings> sceneSettings, GLFWwindow *window) : m_sceneSettings(sceneSettings), m_window(window), m_objects(16){
    m_uniqIdManager = std::make_shared<UniqId>(128);

    m_mainCamera = std::make_shared<Camera>(m_window, m_sceneSettings);
    m_activeCamera = m_mainCamera;
}

void Scene::Init(){
    Add(std::make_shared<Volume3D>());
    Add(std::make_shared<LineGrid>());
}

const std::vector<std::shared_ptr<SceneObject>>& Scene::GetSceneObjects() {
    return m_objects;
}

Scene::~Scene(){
    //do nothing for now
}

int Scene::Add(std::shared_ptr<SceneObject> object, bool active){
    if(object == nullptr) return -1;
    if(object->GetID() != -1) return -1;
    if(m_uniqIdManager->IdExists(object->GetID())) return -2;

    object->SetID(m_uniqIdManager->GetUniqId());
    object->SetActive(active);

    m_objects.push_back(object);
    return object->GetID();
}

void Scene::Remove(int id){
    auto iterator = m_objects.begin();
    for(int i=0; i<m_objects.size(); ++i){
        
        if((*iterator)->GetID() == id) {
            m_objects.erase(iterator);
            break;
        }
    }
}

const std::shared_ptr<Camera>& Scene::GetActiveCam(){
    return m_activeCamera;
}

std::shared_ptr<SceneObject> Scene::Get(int id){
    if(m_uniqIdManager->IdExists(id) == false) return nullptr;

    auto iterator = m_objects.begin();
    for(int i=0; i<m_objects.size(); ++i){
        
        if((*iterator)->GetID() == id) {
            return (*iterator);
        }
    }
    return nullptr;
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
