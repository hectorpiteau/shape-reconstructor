#ifndef OBJECT_LIST_INTERACTOR_H
#define OBJECT_LIST_INTERACTOR_H

#include <memory>

#include "../controllers/Scene/Scene.hpp"
#include "../view/ImUI/ObjectListView.hpp"

#include "SceneObjectInteractor.hpp"

class ObjectListView;

/**
 * @brief The interactor is 
 * 
 */
class ObjectListInteractor {
public:
    ObjectListInteractor(std::shared_ptr<Scene> &scene, std::shared_ptr<ObjectListView> listView, std::shared_ptr<SceneObjectInteractor>& sceneObjectInteractor);
    ObjectListInteractor(const ObjectListInteractor&) = delete;

    ~ObjectListInteractor();
    /**
     * @brief Callback function.
     * 
     * @param active 
     * @param id 
     */
    void SetActive(bool active, int id);

    /**
     * @brief Callback function.
     * 
     * @param id 
     */
    void Select(int id);

    /**
     * @brief Call the UI Renderer and Process the 
     * SceneObject accordingly.
     */
    void Render();

private:

    /**
     * @brief A pointer to the current active scene.
     */
    std::shared_ptr<Scene> m_scene;

    /**
     * @brief A pointer to the main ObjectListView.
     */
    std::shared_ptr<ObjectListView> m_objectListView;
    
    /**
     * @brief SceneObjectInteractor that provide the interface to 
     * interact with SceneObjects.
     */
    std::shared_ptr<SceneObjectInteractor> m_sceneObjectInteractor; 
    
    std::shared_ptr<SceneObject> m_selectedObject;
};

#endif //OBJECT_LIST_INTERACTOR_H