//
// Created by hepiteau on 20/07/23.
//

#ifndef DRTMCS_UNIFORM_DISTRIBUTION_HPP
#define DRTMCS_UNIFORM_DISTRIBUTION_HPP
#include <random>
#include <glm/glm.hpp>

template <typename T>
class UniformDistribution {
private:
    std::default_random_engine generator;
    std::uniform_int_distribution<T> distribution;

public:
    UniformDistribution(T min, T max) : distribution(min, max){}

    T Get(){
        T tmp = distribution(generator);
        return tmp;
    }

};


#endif //DRTMCS_UNIFORM_DISTRIBUTION_HPP
