#include <map>
#include <GLFW/glfw3.h>

class KeyListener
{
private:
    std::map<int, bool> key_pressed;
    
public:
    KeyListener();
    void callback(GLFWwindow *window, int key, int scancode, int action, int mods);

    bool isShiftPressed();
    bool isAltPressed();
    bool isCtrlPressed();
};