//
// Created by hepiteau on 19/07/23.
//

#ifndef DRTMCS_NORMAL_DISTRIBUTION_CLAMPED_H
#define DRTMCS_NORMAL_DISTRIBUTION_CLAMPED_H
#include <iostream>
#include <string>
#include <random>
#include "glm/glm/glm.hpp"

template <typename T>
class NormalDistributionClamped {
private:
    std::default_random_engine generator;
    std::normal_distribution<T> distribution;

public:
    T mean, stddev, clampMin, clampMax;

    NormalDistributionClamped(T mean, T stddev, T clampMin, T clampMax) :
    distribution(mean, stddev),
    mean(mean), stddev(stddev), clampMin(clampMin), clampMax(clampMax){

    }

    ~NormalDistributionClamped() = default;

    NormalDistributionClamped( const NormalDistributionClamped&) = delete;

    T Get(){
        T tmp = distribution(generator);
        return glm::clamp(tmp, clampMin, clampMax);
    }

};


#endif //DRTMCS_NORMAL_DISTRIBUTION_CLAMPED_H
