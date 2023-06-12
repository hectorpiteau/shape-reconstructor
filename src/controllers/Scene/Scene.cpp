#include <vector>
#include <memory>

#include "Scene.hpp"

#include "../../view/SceneObject/SceneObject.hpp"
#include "../../utils/SceneSettings.hpp"
#include "../../model/UniqId.hpp"

Scene::Scene(std::shared_ptr<SceneSettings> sceneSettings, GLFWwindow *window) : m_sceneSettings(sceneSettings),
                                                                                 m_window(window),
                                                                                 m_uniqIdManager(std::make_shared<UniqId>(128)),
                                                                                 m_mainCamera(nullptr),
                                                                                 m_activeCamera(nullptr),
                                                                                 m_objects()
{
    m_mainCamera = std::make_shared<Camera>(this);
    m_activeCamera = m_mainCamera;
    m_objects.push_back(Add(m_mainCamera, true, true));
}

std::vector<std::shared_ptr<SceneObject>> &Scene::GetSceneObjects()
{
    return m_objects;
}
// template<typename T>
std::vector<std::shared_ptr<SceneObject>> Scene::GetAll(SceneObjectTypes type)
{
    std::vector<std::shared_ptr<SceneObject>> tab = std::vector<std::shared_ptr<SceneObject>>();
    for (const auto& obj : m_objects)
    {
        if (obj->GetType() == type)
            tab.push_back(obj);

        std::vector<std::shared_ptr<SceneObject>> tmp = obj->GetAll(type);
        for (const auto& a : tmp)
            tab.push_back(a);
    }
    return tab;
}

std::shared_ptr<SceneObject> Scene::Add(std::shared_ptr<SceneObject> object, bool active, bool isChild)
{
    if (object->GetID() != -1)
        return object;
    if (m_uniqIdManager->IdExists(object->GetID()))
        return object;

    object->SetID(m_uniqIdManager->GetUniqId());
    object->SetActive(active);

    object->SetIsChild(isChild);
    if (!isChild)
        m_objects.push_back(object);
    return object;
}

void Scene::Remove(int id)
{
    auto iterator = m_objects.begin();
    for (long unsigned int i = 0; i < m_objects.size(); ++i)
    {

        if ((*iterator)->GetID() == id)
        {
            m_objects.erase(iterator);
            break;
        }
    }
}

std::shared_ptr<Camera> Scene::GetActiveCam()
{
    return m_activeCamera;
}

std::shared_ptr<Camera> Scene::GetDefaultCam()
{
    return m_mainCamera;
}

std::shared_ptr<SceneObject> Scene::Get(int id)
{
    if (!m_uniqIdManager->IdExists(id))
    {
        std::cout << "SceneObject: " << id << " doest not exist." << std::endl;
        return nullptr;
    }

    for (auto & m_object : m_objects)
    {
        if (m_object->GetID() == id)
        {
            return m_object;
        }
        else
        {
            auto child = m_object->GetChild(id);
            if (child != nullptr)
                return child;
        }
    }
    return nullptr;
}

void Scene::RenderAll()
{
    for (auto &obj : m_objects)
    {
        /** Render only active objects. (The checkbox in the scene object's lists). */
        if (obj != nullptr && obj->IsActive())
        {
            obj->Render();
        }
    }
}

GLFWwindow *Scene::GetWindow()
{
    return m_window;
}
std::shared_ptr<SceneSettings> Scene::GetSceneSettings()
{
    return m_sceneSettings;
}