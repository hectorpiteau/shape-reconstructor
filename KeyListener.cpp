#include "KeyListener.hpp"
#include <iostream>
#include <map>

KeyListener::KeyListener()
{
}

void KeyListener::callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    std::cout << "key pressed: " << key << std::endl;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

    if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL 
    || key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT
    || key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT)
    {
        key_pressed[key] = (action == GLFW_PRESS);
    }
}


bool KeyListener::isAltPressed(){
    return key_pressed[GLFW_KEY_LEFT_ALT] || key_pressed[GLFW_KEY_RIGHT_ALT];
}

bool KeyListener::isShiftPressed(){
    return key_pressed[GLFW_KEY_LEFT_SHIFT] || key_pressed[GLFW_KEY_RIGHT_SHIFT];
}

bool KeyListener::isCtrlPressed(){
    return key_pressed[GLFW_KEY_LEFT_CONTROL] || key_pressed[GLFW_KEY_RIGHT_CONTROL];
}