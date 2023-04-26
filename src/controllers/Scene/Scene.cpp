#include <vector>
#include <memory>

#include "Scene.hpp"

#include "../../view/SceneObject/SceneObject.hpp"
#include "../../utils/SceneSettings.hpp"
#include "../../model/UniqId.hpp"


Scene::Scene(std::shared_ptr<SceneSettings> sceneSettings, GLFWwindow *window) : m_sceneSettings(sceneSettings), m_window(window), m_objects(){
    m_uniqIdManager = std::make_shared<UniqId>(128);

    m_mainCamera = std::make_shared<Camera>(this);
    m_objects.push_back(Add(m_mainCamera, true, true));

    m_activeCamera = m_mainCamera;
}

std::vector<std::shared_ptr<SceneObject>>& Scene::GetSceneObjects() {
    return m_objects;
}

Scene::~Scene(){
    //do nothing for now
}

// template<typename T>
std::vector<std::shared_ptr<SceneObject>> Scene::GetAll(SceneObjectTypes type){
    std::vector<std::shared_ptr<SceneObject>> tab = std::vector<std::shared_ptr<SceneObject>>();
    for(auto obj : m_objects){
        if(obj->GetType() == type) tab.push_back(obj);

        std::vector<std::shared_ptr<SceneObject>> tmp = obj->GetAll(type);
        for(auto a : tmp) tab.push_back(a);
    }
    return tab;
}

std::shared_ptr<SceneObject> Scene::Add(std::shared_ptr<SceneObject> object, bool active, bool isChild){
    if(object->GetID() != -1) return object;
    if(m_uniqIdManager->IdExists(object->GetID())) return object;

    object->SetID(m_uniqIdManager->GetUniqId());
    object->SetActive(active);

    object->SetIsChild(isChild);
    if(isChild == false) m_objects.push_back(object);
    return object;
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

std::shared_ptr<Camera> Scene::GetActiveCam(){
    return m_activeCamera;
}

std::shared_ptr<Camera> Scene::GetDefaultCam(){
    return m_mainCamera;
}

std::shared_ptr<SceneObject> Scene::Get(int id){
    if(m_uniqIdManager->IdExists(id) == false){
        std::cout << "SceneObject: " << id << " doest not exist." << std::endl;
        return nullptr;
    }

    auto iterator = m_objects.begin();

    for(int i=0; i<m_objects.size(); ++i){
        if(m_objects[i]->GetID() == id) {
            return m_objects[i];
        }else{
            auto child = m_objects[i]->GetChild(id);
            if(child != nullptr) return child;
        }
    }
    return nullptr;
}

void Scene::RenderAll()
{
    for(auto &obj : m_objects){
        /** Render only active objects. (The checkbox in the scene object's lists). */
        if(obj != nullptr && obj->IsActive()){
            obj->Render();
        }
    }
}


GLFWwindow *Scene::GetWindow(){
    return m_window;
}
std::shared_ptr<SceneSettings> Scene::GetSceneSettings(){
    return m_sceneSettings;
}