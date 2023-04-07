#pragma once

template<typename T>
class Buffer {
public:
    virtual const T& Get(size_t index) = 0;
    virtual void Set(size_t index, const T& value) = 0;
};