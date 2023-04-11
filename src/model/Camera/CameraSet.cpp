#include <vector>
#include <memory>

#include "Camera.hpp"
#include "CameraSet.hpp"

CameraSet::CameraSet(){

}

CameraSet::~CameraSet()
{

}

size_t CameraSet::Length(){
    return m_cameras.size();
}

void CameraSet::AddCamera(std::shared_ptr<Camera> camera){
    m_cameras.push_back(camera);
}

std::shared_ptr<Camera>& CameraSet::operator[](size_t index){
    return m_cameras[index];
}

std::shared_ptr<Camera> CameraSet::GetCameraById(size_t id){
    for(std::shared_ptr<Camera>& cam : m_cameras){
        if(cam->GetID() == id) return cam;
    }
    return nullptr;
}