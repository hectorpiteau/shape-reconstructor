#pragma once 

#include "../../view/SceneObject/SceneObject.hpp"
#include "../../utils/SceneSettings.hpp"
#include "../../model/Camera.hpp"
#include "../../model/UniqId.hpp"

#include <vector>
#include <memory>

class Scene {
public:
    /**
     * @brief Construct a new Scene object.
     * 
     * @param sceneSettings : A pointer to the sceneSettings object.
     */
    Scene(std::shared_ptr<SceneSettings> sceneSettings, GLFWwindow *window);
    
    /** Delete copy constructor. */
    Scene(const Scene&) = delete;

    /**
     * @brief Destroy the Scene object.
     * 
     */
    ~Scene();

    /**
     * @brief Add an object in the scene. This function updates the object's uniq-id. 
     * With this function, we can only add newly created object, adding the same object 
     * twice will lead to an early return -1.
     * 
     * @param object : The new SceneObject to add to the scene.
     * @param active=true : True if the object is active by default, false to set it directly inactive.
     * @return int : The uniq-id of the newly added object or -1 if not added.
     */
    int Add(std::shared_ptr<SceneObject> object, bool active = true);

    /**
     * @brief Get a SceneObject from it's id.
     * 
     * @param id : The id of the SceneObject to get.
     * @return std::shared_ptr<SceneObject> : The SceneObject if it exist, nullptr otherwise.
     */
    std::shared_ptr<SceneObject> Get(int id);
    
    /**
     * @brief Remove an object from the scene.
     * 
     * @param id 
     */
    void Remove(int id);

    /**
     * @brief Render all active elements in the scene.
     */
    void RenderAll();

    /**
     * @brief Get the currently active camera.
     * 
     * @return const Camera& : A constant ref to the active camera.
     */
    const std::shared_ptr<Camera>& GetActiveCam();

    const std::vector<std::shared_ptr<SceneObject>>& GetSceneObjects();

    void Init();

private:
    /**
     * @brief Current Window.
     * 
     */
    GLFWwindow *m_window;
    
    /**
     * @brief A manager that handle uniq-ids.
     */
    std::shared_ptr<UniqId> m_uniqIdManager;

    /**
     * @brief The main camera is the default that is used to 
     * visualise the scene.
     */
    std::shared_ptr<Camera> m_mainCamera;

    /**
     * @brief The active camera is a pointer to another camera.
     * By default it points to the mainCamera, but it can point
     * to another camera.
     */
    std::shared_ptr<Camera> m_activeCamera;

    /**
     * @brief The scene's settings object.
     * 
     */
    std::shared_ptr<SceneSettings> m_sceneSettings;

    /**
     * @brief The list of objects in the scene. 
     */
    std::vector<std::shared_ptr<SceneObject>> m_objects;    
    
};