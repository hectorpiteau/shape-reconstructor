#pragma once 
#include "../Renderable/Renderable.hpp"
#include <string>

class SceneObject : public Renderable{
public:
    int GetID(){ return m_id;}
    int SetID(int id) { m_id = id;}

    bool IsActive(){ return m_active;}
    void SetActive(bool active) { m_active = active; }

    void SetName(const std::string &name){ m_name = name;}
    const std::string& GetName(){ return m_name;}

private:
    /** A uniq-id that identify this SceneObject. */
    int m_id;
    /** True if the object is active in the scene (visible) or not. */
    bool m_active;
    /** The Object's name. */
    std::string m_name;
};