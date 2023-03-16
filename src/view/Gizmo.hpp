#ifndef GYZMO_H
#define GYZMO_H
#include <glm/glm.hpp>

class Gyzmo {
public:
    Gyzmo();
    void Render();

private:
    float mXLength;
    float mYLength;
    float mZLength;

    glm::vec4 mXColor;
    glm::vec4 mYColor;
    glm::vec4 mZColor;
};

#endif //GYZMO_H