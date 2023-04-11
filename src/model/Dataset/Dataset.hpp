#pragma once
#include <string>

/**
 * @brief 
 * 
 */
class Dataset {
public:

Dataset(const std::string& name) : m_name(name) {

};

const std::string& GetName(){
    return m_name;
}

/**
 * @brief Load the dataset files in host memory. 
 * 
 * @return true : Loaded with success.
 * @return false : Not loaded.
 */
virtual bool Load() = 0;

virtual size_t Size() = 0;



private:
    std::string m_name;
};